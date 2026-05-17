"""Prophet を使った売上予測モデル"""
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics


def detect_closed_days(sales_df: pd.DataFrame, threshold: float = 0.7) -> dict:
    """
    過去データから休業パターンを分析して返す。
    戻り値:
      closed_weekdays: 休業率が threshold 以上の曜日番号のセット（0=月〜6=日）
      closed_dates:    特定日付の臨時休業セット
      stats:           曜日別の休業率テーブル（表示用）
    """
    df = sales_df.copy()
    df["is_closed"] = (df["y"] == 0).astype(int)
    df["weekday"] = df["ds"].dt.dayofweek

    stats = df.groupby("weekday").agg(
        total=("is_closed", "count"),
        closed=("is_closed", "sum"),
    ).reset_index()
    stats["closed_rate"] = stats["closed"] / stats["total"]

    dow_labels = {0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"}
    stats["曜日"] = stats["weekday"].map(dow_labels)
    stats["休業率"] = (stats["closed_rate"] * 100).round(1).astype(str) + "%"

    closed_weekdays = set(stats.loc[stats["closed_rate"] >= threshold, "weekday"].tolist())

    # 定休日パターンに当てはまらない0円日 → 臨時休業
    regular_closed_mask = df["weekday"].isin(closed_weekdays)
    closed_dates = set(df.loc[(df["is_closed"] == 1) & (~regular_closed_mask), "ds"].dt.date)

    return {
        "closed_weekdays": closed_weekdays,
        "closed_dates": closed_dates,
        "stats": stats[["曜日", "total", "closed", "休業率"]].rename(
            columns={"total": "営業日数+休業日数", "closed": "休業日数"}
        ),
    }


def apply_closed_days(forecast: pd.DataFrame, closed_info: dict) -> pd.DataFrame:
    """予測結果の休業日を0円にする"""
    df = forecast.copy()
    df["weekday"] = df["ds"].dt.dayofweek
    df["date"] = df["ds"].dt.date

    is_regular_closed = df["weekday"].isin(closed_info["closed_weekdays"])
    is_spot_closed = df["date"].isin(closed_info["closed_dates"])
    is_closed = is_regular_closed | is_spot_closed

    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        if col in df.columns:
            df.loc[is_closed, col] = 0

    df["is_closed"] = is_closed.astype(int)
    return df


def build_features(
    sales_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    trends_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """売上 + 天候 + トレンド + 祝日特徴量を結合する（休業日は除外）"""
    from src.holidays import add_holiday_features

    df = sales_df.copy()
    # 0円（休業日）を除外してから学習する
    df = df[df["y"] > 0].copy()
    df = df.merge(
        weather_df[["ds", "temp_avg", "is_rain", "is_snow", "precipitation"]],
        on="ds",
        how="left",
    )
    if trends_df is not None and not trends_df.empty:
        df = df.merge(trends_df[["ds", "trends_index"]], on="ds", how="left")
    df = add_holiday_features(df)  # is_obon フラグを含む
    df = df.dropna(subset=["y"])
    # リグレッサーの欠損を中央値で補完
    for col in ["temp_avg", "is_rain", "is_snow", "precipitation", "trends_index"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


def train_model(
    df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    changepoint_prior_scale: float = 0.01,
    sales_cap: float = None,
    sales_floor: float = None,
) -> Prophet:
    use_logistic = sales_cap is not None

    if use_logistic:
        df = df.copy()
        df["cap"] = float(sales_cap)
        df["floor"] = float(sales_floor) if sales_floor else 0.0

    model = Prophet(
        holidays=holidays_df,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=changepoint_prior_scale,
        changepoint_range=0.9,
        holidays_prior_scale=10.0,
        growth="logistic" if use_logistic else "linear",
    )
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    # 天候リグレッサー（データがある場合のみ）
    if "temp_avg" in df.columns and df["temp_avg"].notna().sum() > 30:
        model.add_regressor("temp_avg", standardize=True)
    if "is_rain" in df.columns and df["is_rain"].notna().sum() > 30:
        model.add_regressor("is_rain", standardize=False)
    if "is_snow" in df.columns and df["is_snow"].notna().sum() > 10:
        model.add_regressor("is_snow", standardize=False)

    # お盆リグレッサー（公式祝日ではないが売上に影響）
    if "is_obon" in df.columns and df["is_obon"].sum() > 0:
        model.add_regressor("is_obon", standardize=False)

    # Google Trends リグレッサー
    if "trends_index" in df.columns and df["trends_index"].notna().sum() > 30:
        model.add_regressor("trends_index", standardize=True)

    regressor_cols = [c for c in ["temp_avg", "is_rain", "is_snow", "is_obon", "trends_index"] if c in df.columns]
    cols = ["ds", "y"] + regressor_cols
    if use_logistic:
        cols += ["cap", "floor"]
    train_df = df[cols].copy()
    train_df = train_df.dropna(subset=["y"])

    model.fit(train_df)
    return model, use_logistic, sales_cap, sales_floor


def make_future_df(
    model: Prophet,
    future_weather_df: pd.DataFrame,
    periods: int = 365,
    use_logistic: bool = False,
    sales_cap: float = None,
    sales_floor: float = None,
    future_trends_df: pd.DataFrame = None,
) -> pd.DataFrame:
    # model.make_future_dataframe は学習データ（休業日除外）の日付のみを返すため
    # 全カレンダー日付を含む連続した日付範囲で予測フレームを生成する
    train_start = model.history["ds"].min()
    train_end = model.history["ds"].max()
    forecast_end = train_end + pd.Timedelta(days=periods)
    all_dates = pd.date_range(start=train_start, end=forecast_end, freq="D")
    future = pd.DataFrame({"ds": all_dates})
    future = future.merge(
        future_weather_df[["ds", "temp_avg", "is_rain", "is_snow"]],
        on="ds",
        how="left",
    )
    for col in ["temp_avg", "is_rain", "is_snow"]:
        if col in future.columns:
            future[col] = future[col].fillna(future[col].mean())

    if future_trends_df is not None and not future_trends_df.empty:
        future = future.merge(future_trends_df[["ds", "trends_index"]], on="ds", how="left")
        future["trends_index"] = future["trends_index"].fillna(future["trends_index"].mean())

    # お盆フラグ（8/10〜8/18）
    future["is_obon"] = future["ds"].apply(
        lambda d: int(d.month == 8 and 10 <= d.day <= 18)
    )

    if use_logistic:
        future["cap"] = float(sales_cap)
        future["floor"] = float(sales_floor) if sales_floor else 0.0
    return future


def adjust_by_monthly_baseline(
    forecast_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    base_year: int,
    growth_rate: float = 0.0,
) -> pd.DataFrame:
    """
    前年同月の実績合計を基準に、予測の月次合計をスケール補正する。
    曜日・祝日の日内分布パターンはProphetのものをそのまま維持する。
    """
    df = forecast_df.copy()

    # 前年の月次合計（営業日のみ）
    base = actual_df[(actual_df["ds"].dt.year == base_year) & (actual_df["y"] > 0)].copy()
    monthly_base = base.groupby(base["ds"].dt.month)["y"].sum()

    result_rows = []
    for month in range(1, 13):
        month_mask = df["ds"].dt.month == month
        month_df = df[month_mask].copy()

        if month not in monthly_base.index:
            # 前年データがない月はそのまま
            result_rows.append(month_df)
            continue

        target_total = monthly_base[month] * (1 + growth_rate)
        open_mask = month_df["yhat"] > 0
        current_total = month_df.loc[open_mask, "yhat"].sum()

        if current_total > 0:
            scale = target_total / current_total
            for col in ["yhat", "yhat_lower", "yhat_upper"]:
                if col in month_df.columns:
                    month_df.loc[open_mask, col] = (month_df.loc[open_mask, col] * scale).clip(lower=0)

        result_rows.append(month_df)

    return pd.concat(result_rows).sort_values("ds").reset_index(drop=True)


def adjust_by_manual_monthly_avg(
    forecast_df: pd.DataFrame,
    manual_monthly_avg: dict,
    growth_rate: float = 0.0,
) -> pd.DataFrame:
    """
    手動入力された月次営業日平均を基準にスケール補正する。
    {月: 営業日平均} × (1 + growth_rate) が各月の目標平均になる。
    """
    df = forecast_df.copy()
    # is_closed列があれば営業日フラグを使用（Prophet予測がマイナスでも正しく処理）
    if "is_closed" in df.columns:
        open_mask = df["is_closed"] == 0
    else:
        open_mask = df["yhat"] > 0

    for month, base_avg in manual_monthly_avg.items():
        m_open = (df["ds"].dt.month == month) & open_mask
        if m_open.sum() == 0:
            continue
        target_avg = base_avg * (1 + growth_rate)
        current_avg = df.loc[m_open, "yhat"].mean()
        if current_avg <= 0:
            # Prophetがその月全体をゼロ以下と予測した場合は直接目標値を設定
            df.loc[m_open, "yhat"] = target_avg
            if "yhat_lower" in df.columns:
                df.loc[m_open, "yhat_lower"] = target_avg * 0.8
            if "yhat_upper" in df.columns:
                df.loc[m_open, "yhat_upper"] = target_avg * 1.2
            continue
        scale = target_avg / current_avg
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            if col in df.columns:
                df.loc[m_open, col] = (df.loc[m_open, col] * scale).clip(lower=0)

    return df


def apply_obon_boost(
    forecast_df: pd.DataFrame,
    obon_multiplier: float = 1.8,
    obon_start_day: int = 10,
    obon_end_day: int = 18,
) -> pd.DataFrame:
    """
    お盆期間（8月 obon_start_day〜obon_end_day）の予測を底上げする。
    月合計は変えず、お盆日に重みをつけて再配分する。
    multiplier=1.8 → お盆日が通常日の1.8倍の重みになる。
    """
    df = forecast_df.copy()

    for year in df["ds"].dt.year.unique():
        aug_mask = (df["ds"].dt.year == year) & (df["ds"].dt.month == 8) & (df["yhat"] > 0)
        obon_mask = aug_mask & (df["ds"].dt.day >= obon_start_day) & (df["ds"].dt.day <= obon_end_day)
        non_obon_mask = aug_mask & ~(df["ds"].dt.day >= obon_start_day) & ~(df["ds"].dt.day > obon_end_day)
        non_obon_mask = aug_mask & (
            (df["ds"].dt.day < obon_start_day) | (df["ds"].dt.day > obon_end_day)
        )

        if obon_mask.sum() == 0 or non_obon_mask.sum() == 0:
            continue

        # 現在の8月合計を保持
        aug_total = df.loc[aug_mask, "yhat"].sum()

        # 重みつき再配分
        n_obon = obon_mask.sum()
        n_non = non_obon_mask.sum()
        # total_weight = n_obon * multiplier + n_non * 1.0
        total_weight = n_obon * obon_multiplier + n_non * 1.0
        unit = aug_total / total_weight

        df.loc[obon_mask, "yhat"] = df.loc[obon_mask, "yhat"] / df.loc[obon_mask, "yhat"].mean() * (unit * obon_multiplier)
        df.loc[non_obon_mask, "yhat"] = df.loc[non_obon_mask, "yhat"] / df.loc[non_obon_mask, "yhat"].mean() * unit

        for col in ["yhat_lower", "yhat_upper"]:
            if col in df.columns:
                ratio = df.loc[aug_mask, "yhat"] / df.loc[aug_mask, "yhat"].clip(lower=1)
                df.loc[aug_mask, col] = (df.loc[aug_mask, col] * ratio).clip(lower=0)

    return df


def apply_min_daily_floor(
    forecast_df: pd.DataFrame,
    min_monthly_avg: float,
    min_annual_avg: float,
) -> pd.DataFrame:
    """
    営業日平均の下限を保証する（個々の日の値は変えない）。
    ① 月次：月間営業日平均が min_monthly_avg を下回る月をスケールアップ
    ② 年間：全体の営業日平均が min_annual_avg を下回る場合、全月を一律スケールアップ
    """
    df = forecast_df.copy()
    # is_closed列があれば営業日フラグを使用（Prophet予測がマイナスでも正しく処理）
    if "is_closed" in df.columns:
        open_mask = df["is_closed"] == 0
    else:
        open_mask = df["yhat"] > 0

    # ① 月次下限
    for month in range(1, 13):
        m_open = (df["ds"].dt.month == month) & open_mask
        if m_open.sum() == 0:
            continue
        monthly_avg = df.loc[m_open, "yhat"].mean()
        if monthly_avg < min_monthly_avg:
            if monthly_avg <= 0:
                df.loc[m_open, "yhat"] = min_monthly_avg
            else:
                scale = min_monthly_avg / monthly_avg
                for col in ["yhat", "yhat_lower", "yhat_upper"]:
                    if col in df.columns:
                        df.loc[m_open, col] *= scale

    # ② 年間下限（月次補正後に再チェック）
    annual_avg = df.loc[open_mask, "yhat"].mean()
    if annual_avg < min_annual_avg and annual_avg > 0:
        scale = min_annual_avg / annual_avg
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            if col in df.columns:
                df.loc[open_mask, col] *= scale

    # ③ 日別の絶対下限（マイナスや極端に低い値を防ぐ）
    df.loc[open_mask, "yhat"] = df.loc[open_mask, "yhat"].clip(lower=8_000)
    if "yhat_lower" in df.columns:
        df.loc[open_mask, "yhat_lower"] = df.loc[open_mask, "yhat_lower"].clip(lower=8_000)

    return df


def apply_weekday_weekend_correction(
    forecast_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    forecast_year: int,
) -> tuple:
    """
    実績データから「土日祝 vs 平日」の乖離パターンを学習し、
    月次合計を変えずに日別を再配分する。
    戻り値: (補正済みDataFrame, 土日祝補正係数, 平日補正係数)
    実績が20件未満の場合は補正なしで元のDataFrameを返す。
    """
    from datetime import date as date_type

    df = forecast_df.copy()
    today = date_type.today()

    # 今年の確定実績（本日以前・売上>0）
    actual = actual_df[
        (actual_df["ds"].dt.year == forecast_year) &
        (actual_df["y"] > 0) &
        (actual_df["ds"].dt.date < today)
    ].copy()

    if len(actual) < 20:
        return df, None, None

    # 祝日セット
    holiday_dates = set(holidays_df["ds"].dt.date)

    def is_wh(ds):
        return ds.dayofweek >= 5 or ds.date() in holiday_dates

    # 実績の土日祝 / 平日 平均
    actual["_wh"] = actual["ds"].apply(is_wh)
    actual_wh_avg = actual.loc[actual["_wh"], "y"].mean()
    actual_wd_avg = actual.loc[~actual["_wh"], "y"].mean()

    # 同期間の予測値
    pred_same = df[
        df["ds"].isin(actual["ds"].values) & (df["is_closed"] == 0)
    ].copy()
    pred_same["_wh"] = pred_same["ds"].apply(is_wh)
    pred_wh_avg = pred_same.loc[pred_same["_wh"], "yhat"].mean()
    pred_wd_avg = pred_same.loc[~pred_same["_wh"], "yhat"].mean()

    if pred_wh_avg <= 0 or pred_wd_avg <= 0 or actual_wd_avg <= 0:
        return df, None, None

    wh_factor = actual_wh_avg / pred_wh_avg  # > 1 なら実績が予測より高い
    wd_factor = actual_wd_avg / pred_wd_avg  # < 1 なら実績が予測より低い

    # 月ごとに月次合計を保ちながら再配分
    open_mask = df["is_closed"] == 0
    df["_wh"] = df["ds"].apply(is_wh)

    for month in range(1, 13):
        m_open = (df["ds"].dt.month == month) & open_mask
        m_wh = m_open & df["_wh"]
        m_wd = m_open & ~df["_wh"]

        if m_wh.sum() == 0 or m_wd.sum() == 0:
            continue

        current_total = df.loc[m_open, "yhat"].sum()
        if current_total <= 0:
            continue

        new_wh = df.loc[m_wh, "yhat"].sum() * wh_factor
        new_wd = df.loc[m_wd, "yhat"].sum() * wd_factor
        new_total = new_wh + new_wd
        if new_total <= 0:
            continue

        scale = current_total / new_total
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            if col in df.columns:
                df.loc[m_wh, col] = (df.loc[m_wh, col] * wh_factor * scale).clip(lower=0)
                df.loc[m_wd, col] = (df.loc[m_wd, col] * wd_factor * scale).clip(lower=0)

    df = df.drop(columns=["_wh"])
    return df, wh_factor, wd_factor


def get_monthly_baseline_table(actual_df: pd.DataFrame, base_year: int) -> pd.DataFrame:
    """前年の月次実績サマリーを返す（サイドバー表示用）"""
    base = actual_df[(actual_df["ds"].dt.year == base_year) & (actual_df["y"] > 0)].copy()
    if base.empty:
        return pd.DataFrame()

    tbl = base.groupby(base["ds"].dt.month).agg(
        月次合計=("y", "sum"),
        営業日数=("y", "count"),
    ).reset_index()
    tbl.columns = ["月", "月次合計", "営業日数"]
    tbl["日平均"] = (tbl["月次合計"] / tbl["営業日数"]).round(0).astype(int)
    return tbl


def evaluate_model(model: Prophet, df: pd.DataFrame) -> pd.DataFrame:
    """クロスバリデーションで精度評価する"""
    data_days = (df["ds"].max() - df["ds"].min()).days
    if data_days < 365:
        return pd.DataFrame()

    horizon = "90 days"
    initial = f"{max(180, data_days // 2)} days"
    period = "60 days"

    cv_df = cross_validation(model, initial=initial, period=period, horizon=horizon, parallel="threads")
    metrics_df = performance_metrics(cv_df)
    return metrics_df


def get_component_contributions(forecast: pd.DataFrame) -> dict:
    """予測の構成要素を返す"""
    components = {}
    for col in ["trend", "weekly", "yearly", "monthly", "holidays"]:
        if col in forecast.columns:
            components[col] = forecast[col]
    return components
