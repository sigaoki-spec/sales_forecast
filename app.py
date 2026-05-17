"""売上予測アプリ - Streamlit メインファイル"""
import os
import sys
from datetime import date, timedelta

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_from_google_sheets, load_from_google_sheets_horizontal, load_from_csv, generate_sample_data, write_forecast_to_sheets
from src.trends import fetch_google_trends, estimate_future_trends
from src.weather import fetch_historical_weather, fetch_forecast_weather, estimate_future_weather_from_history
from src.holidays import add_holiday_features, build_prophet_holidays
from src.model import (build_features, train_model, make_future_df, evaluate_model,
                       detect_closed_days, apply_closed_days,
                       adjust_by_manual_monthly_avg, apply_weekday_weekend_correction,
                       apply_obon_boost, apply_min_daily_floor, get_monthly_baseline_table)

st.set_page_config(
    page_title="売上予測ダッシュボード",
    page_icon="📊",
    layout="wide",
)

# ─── サイドバー設定 ────────────────────────────────────────────────
st.sidebar.title("⚙️ 設定")

data_source = st.sidebar.radio(
    "データソース",
    ["Google スプレッドシート", "CSV ファイル", "サンプルデータで試す"],
)

if data_source == "Google スプレッドシート":
    spreadsheet_id = st.sidebar.text_input(
        "スプレッドシート ID",
        value=os.getenv("SPREADSHEET_ID", ""),
        help="URLの /d/XXXXX/edit の XXXXX 部分",
    )
    cred_path = st.sidebar.text_input(
        "サービスアカウント JSON パス",
        value=os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH", "credentials.json"),
    )
    sheet_name = st.sidebar.text_input("シート名（空欄で先頭シート）", value="")

    layout_type = st.sidebar.radio(
        "データの並び方",
        ["縦（日付と売上が列）", "横（日付と売上が行）"],
        index=1,
        help="縦：A列=日付・B列=売上 / 横：2行目=日付・24行目=総売上 のような配置",
    )

    if layout_type == "縦（日付と売上が列）":
        date_col_idx = st.sidebar.number_input("日付の列番号（A列=0）", value=0, min_value=0)
        sales_col_idx = st.sidebar.number_input("売上の列番号（B列=1）", value=1, min_value=0)
        header_row = st.sidebar.number_input("ヘッダー行数", value=1, min_value=0)
    else:
        date_row_idx = st.sidebar.number_input("日付の行番号", value=2, min_value=1,
                                               help="スプレッドシートの行番号をそのまま入力")
        sales_row_idx = st.sidebar.number_input("売上の行番号", value=30, min_value=1,
                                                help="スプレッドシートの行番号をそのまま入力")
        start_col_idx = st.sidebar.number_input("データ開始列番号（A列=0）", value=4, min_value=0,
                                                help="E列から始まる場合は4")

elif data_source == "CSV ファイル":
    uploaded_file = st.sidebar.file_uploader("CSV ファイルをアップロード", type=["csv"])
    date_col_name = st.sidebar.text_input("日付列名", value="date")
    sales_col_name = st.sidebar.text_input("売上列名", value="sales")

st.sidebar.markdown("---")
st.sidebar.subheader("📍 店舗の場所（天候取得用）")
location_name = st.sidebar.text_input("地名", value=os.getenv("LOCATION_NAME", "東京"))
lat = st.sidebar.number_input("緯度", value=float(os.getenv("LATITUDE", 35.6762)), format="%.4f")
lon = st.sidebar.number_input("経度", value=float(os.getenv("LONGITUDE", 139.6503)), format="%.4f")

st.sidebar.markdown("---")
st.sidebar.subheader("📅 予測設定")
forecast_year = st.sidebar.number_input("予測対象年", value=date.today().year, min_value=2024, max_value=2030)
base_year = forecast_year - 1  # 前年

st.sidebar.markdown("---")
st.sidebar.subheader("📈 前年比・月次実績設定")
growth_rate = 0.0
manual_monthly_avg = {}  # {月: 営業日平均売上}

growth_pct = st.sidebar.slider(
    f"前年（{base_year}年）比",
    min_value=-30, max_value=50, value=10, step=1,
    format="%d%%",
    help="0%=前年と同じ、+10%=前年より10%増を想定",
)
growth_rate = growth_pct / 100.0
if growth_pct > 0:
    st.sidebar.success(f"前年比 +{growth_pct}%（増収想定）")
elif growth_pct < 0:
    st.sidebar.warning(f"前年比 {growth_pct}%（減収想定）")
else:
    st.sidebar.info("前年並みを想定")

st.sidebar.caption("各月の「営業日あたりの売上平均」（前年実績）")
# デフォルト値（前年実績をもとに設定）
default_avgs = {
    1: 33199, 2: 33678, 3: 34838,
    4: 35974, 5: 36819, 6: 26304,
    7: 35529, 8: 54293, 9: 40670,
    10: 31767, 11: 33701, 12: 26900,
}
month_names = {1:"1月",2:"2月",3:"3月",4:"4月",5:"5月",6:"6月",
               7:"7月",8:"8月",9:"9月",10:"10月",11:"11月",12:"12月"}
cols_a, cols_b = st.sidebar.columns(2)
for m in range(1, 13):
    col = cols_a if m % 2 == 1 else cols_b
    val = col.number_input(
        month_names[m], value=default_avgs[m],
        step=500, key=f"manual_avg_{m}",
    )
    manual_monthly_avg[m] = val

st.sidebar.markdown("---")
st.sidebar.subheader("🏮 お盆ウィーク設定（8月）")
obon_multiplier = st.sidebar.slider(
    "お盆期間（8/10〜18）の売上倍率",
    min_value=1.0, max_value=3.0, value=1.8, step=0.1,
    help="通常日を1.0とした場合のお盆期間の売上比率。8月の月合計は変わりません。",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ 営業日平均の下限設定")
min_monthly_avg = st.sidebar.number_input(
    "月間 営業日平均の下限（円）",
    value=26_000,
    step=1_000,
    help="月間の営業日平均がこの金額を下回る月は底上げします。個々の日は下回ることがあります。",
)
min_annual_avg = st.sidebar.number_input(
    "年間 営業日平均の下限（円）",
    value=35_000,
    step=1_000,
    help="年間の営業日平均がこの金額を下回る場合、全体を底上げします。",
)
st.sidebar.caption("個々の日の売上は変動します（¥8,000台の日も許容）")

# モデルパラメータ（詳細設定）
with st.sidebar.expander("⚙️ 詳細設定"):
    changepoint_prior = st.slider(
        "トレンド感度", min_value=0.001, max_value=0.1,
        value=0.01, step=0.001, format="%.3f",
    )
    use_sales_cap = st.checkbox("売上の上限・下限を設定する", value=False)
    sales_cap = sales_floor = None
    if use_sales_cap:
        sales_cap = st.number_input("1日の売上上限（円）", value=500_000, step=10_000)
        sales_floor = st.number_input("1日の売上下限（円）", value=30_000, step=5_000)

run_button = st.sidebar.button("🔮 予測を実行", type="primary", use_container_width=True)

if data_source == "Google スプレッドシート":
    st.sidebar.markdown("---")
    st.sidebar.caption("23行目:予測 / 24行目:下限 / 25行目:上限")
    if st.sidebar.button("📤 スプレッドシートに書き込む", use_container_width=True):
        if "forecast_cache" not in st.session_state:
            st.sidebar.warning("先に「予測を実行」を押してください。")
        else:
            _fc = st.session_state["forecast_cache"]
            try:
                result = write_forecast_to_sheets(
                    spreadsheet_id=spreadsheet_id,
                    credentials_path=cred_path,
                    forecast_df=_fc["forecast_year_df"],
                    sheet_name=sheet_name or None,
                    date_row=date_row_idx,
                    target_row=23,
                    start_col=start_col_idx,
                )
                if result["written"] > 0:
                    st.sidebar.success(f"✅ {result['written']}件 書き込み完了")
                else:
                    st.sidebar.warning(
                        f"⚠️ 0件でした\n"
                        f"日付サンプル: {', '.join(result['date_sample'])}\n"
                        f"有効列数: {result['total_cols']}"
                    )
            except Exception as e:
                st.sidebar.error(f"エラー: {e}")


# ─── メインエリア ────────────────────────────────────────────────
st.title("📊 飲食店 売上予測ダッシュボード")
st.caption(f"曜日・祝日・天候・気温を加味した日次売上予測 | 予測対象: {forecast_year}年")

if not run_button and "forecast_cache" not in st.session_state:
    st.info("👈 左のサイドバーで設定を行い、「予測を実行」ボタンを押してください。")
    st.markdown("""
    ### このアプリでできること
    - 過去の日次売上データを読み込み
    - **曜日**・**祝日**・**天候**・**気温**の影響を学習
    - 今年1年間の日次売上を予測
    - 売上に影響する要因の分解グラフを表示
    - 月次・週次サマリーの表示

    ### データ形式（Google スプレッドシート / CSV）
    | 日付 | 売上 |
    |------|------|
    | 2024-01-01 | 150000 |
    | 2024-01-02 | 230000 |
    | ... | ... |
    """)
    st.stop()

# ─── キャッシュ読み込み or 新規計算 ─────────────────────────────────
if not run_button:
    _c = st.session_state["forecast_cache"]
    forecast_year_df        = _c["forecast_year_df"]
    comparison_df           = _c["comparison_df"]
    has_actual              = _c["has_actual"]
    actual_in_forecast_year = _c["actual_in_forecast_year"]
    sales_df                = _c["sales_df"]
    historical_weather      = _c["historical_weather"]
    forecast                = _c["forecast"]
    holidays_df             = _c["holidays_df"]
    closed_info             = _c["closed_info"]
    corrected_months        = _c["corrected_months"]
else:
    corrected_months = []

    # ─── データ読み込み ─────────────────────────────────────────────
    with st.spinner("データを読み込んでいます..."):
        try:
            if data_source == "Google スプレッドシート":
                if not spreadsheet_id:
                    st.error("スプレッドシート ID を入力してください。")
                    st.stop()
                if layout_type == "縦（日付と売上が列）":
                    sales_df = load_from_google_sheets(
                        spreadsheet_id=spreadsheet_id,
                        credentials_path=cred_path,
                        sheet_name=sheet_name or None,
                        date_col=date_col_idx,
                        sales_col=sales_col_idx,
                        header_row=header_row,
                    )
                else:
                    sales_df = load_from_google_sheets_horizontal(
                        spreadsheet_id=spreadsheet_id,
                        credentials_path=cred_path,
                        sheet_name=sheet_name or None,
                        date_row=date_row_idx,
                        sales_row=sales_row_idx,
                        start_col=start_col_idx,
                    )
            elif data_source == "CSV ファイル":
                if uploaded_file is None:
                    st.error("CSV ファイルをアップロードしてください。")
                    st.stop()
                sales_df = load_from_csv(uploaded_file, date_col=date_col_name, sales_col=sales_col_name)
            else:
                sales_df = generate_sample_data(years=3)
                st.info("サンプルデータを使用しています（過去3年分）。")

            total_days = len(sales_df)
            closed_days = (sales_df["y"] == 0).sum()
            open_days = total_days - closed_days
            st.success(
                f"売上データ読み込み完了: {total_days}日分（{sales_df['ds'].min().date()} 〜 {sales_df['ds'].max().date()}）"
                f" ／ 営業日: {open_days}日・休業日: {closed_days}日"
            )

            baseline_tbl = get_monthly_baseline_table(sales_df, base_year)
            if not baseline_tbl.empty:
                st.sidebar.markdown(f"**{base_year}年 月次実績（参考）**")
                disp = baseline_tbl.copy()
                disp = disp.rename(columns={
                    "月次合計": "売上",
                    "日平均": "営業日売上平均",
                })
                disp["売上"] = disp["売上"].map(lambda x: f"¥{x:,.0f}")
                disp["営業日売上平均"] = disp["営業日売上平均"].map(lambda x: f"¥{x:,.0f}")
                st.sidebar.dataframe(disp.set_index("月"), use_container_width=True)

        except Exception as e:
            st.error(f"データ読み込みエラー: {e}")
            st.stop()

    # ─── 休業日パターン分析 ─────────────────────────────────────────
    closed_info = detect_closed_days(sales_df)
    if closed_info["closed_weekdays"]:
        dow_labels = {0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"}
        closed_names = "・".join(dow_labels[d] for d in sorted(closed_info["closed_weekdays"]))
        st.info(f"🔍 定休日を検出しました：**{closed_names}曜日**　（予測でも休業日として0円に設定します）")

    # ─── 天候データ取得 ─────────────────────────────────────────────
    with st.spinner("天候データを取得しています..."):
        try:
            hist_start = sales_df["ds"].min().date()
            hist_end = min(sales_df["ds"].max().date(), date.today() - timedelta(days=2))

            historical_weather = fetch_historical_weather(lat, lon, hist_start, hist_end)

            forecast_end = date(forecast_year, 12, 31)
            forecast_weather_near = fetch_forecast_weather(lat, lon, days=16)
            near_end = forecast_weather_near["ds"].max().date()

            if near_end < forecast_end:
                far_dates = pd.date_range(
                    start=near_end + timedelta(days=1),
                    end=forecast_end,
                    freq="D",
                )
                far_weather = estimate_future_weather_from_history(historical_weather, far_dates)
                all_future_weather = pd.concat([forecast_weather_near, far_weather], ignore_index=True)
            else:
                all_future_weather = forecast_weather_near

            full_weather = pd.concat([historical_weather, all_future_weather], ignore_index=True)
            full_weather = full_weather.drop_duplicates(subset=["ds"]).sort_values("ds").reset_index(drop=True)

            st.success(f"天候データ取得完了: {location_name}（{hist_start} 〜 {forecast_end}）")
        except Exception as e:
            st.warning(f"天候データの取得に失敗しました: {e}\n天候なしで予測します。")
            full_weather = pd.DataFrame(columns=["ds", "temp_avg", "is_rain", "is_snow", "precipitation"])
            historical_weather = full_weather.copy()

    # ─── Google Trends 取得 ──────────────────────────────────────────
    historical_trends = pd.DataFrame()
    future_trends_df = pd.DataFrame()
    with st.spinner("Google Trends（館山 検索量）を取得しています..."):
        try:
            trends_start = sales_df["ds"].min().date()
            trends_end = date.today()
            historical_trends = fetch_google_trends(start_date=trends_start, end_date=trends_end)
            future_dates_trends = pd.date_range(
                start=pd.Timestamp(trends_end),
                end=date(forecast_year, 12, 31),
                freq="D",
            )
            future_trends_df = estimate_future_trends(historical_trends, future_dates_trends)
            full_trends = pd.concat([historical_trends, future_trends_df], ignore_index=True)
            full_trends = full_trends.drop_duplicates(subset=["ds"]).sort_values("ds").reset_index(drop=True)
            st.success(f"Google Trends 取得完了（{len(historical_trends)}日分）")
        except Exception as e:
            st.warning(f"Google Trends の取得に失敗しました: {e}\nトレンドなしで予測します。")
            full_trends = pd.DataFrame(columns=["ds", "trends_index"])

    # ─── 特徴量エンジニアリング & モデル学習 ─────────────────────────
    with st.spinner("予測モデルを学習しています..."):
        try:
            feature_df = build_features(sales_df, full_weather, trends_df=full_trends)
            holidays_df = build_prophet_holidays()
            model, use_logistic, _cap, _floor = train_model(
                feature_df, holidays_df,
                changepoint_prior_scale=changepoint_prior,
                sales_cap=sales_cap if use_sales_cap else None,
                sales_floor=sales_floor if use_sales_cap else None,
            )
            st.success("モデル学習完了！")
        except Exception as e:
            st.error(f"モデル学習エラー: {e}")
            st.stop()

    # ─── 予測実行 ───────────────────────────────────────────────────
    with st.spinner("予測を計算しています..."):
        last_training_date = feature_df["ds"].max().date()
        periods = (date(forecast_year, 12, 31) - last_training_date).days + 1
        periods = max(periods, 1)

        future_df = make_future_df(
            model, full_weather, periods=periods,
            use_logistic=use_logistic,
            sales_cap=_cap,
            sales_floor=_floor,
            future_trends_df=full_trends,
        )
        forecast = model.predict(future_df)

        today_date = date.today()
        past_closed_dates = {d for d in closed_info["closed_dates"] if d < today_date}
        closed_info_for_forecast = {
            "closed_weekdays": set(),
            "closed_dates": past_closed_dates,
        }
        forecast = apply_closed_days(forecast, closed_info_for_forecast)

        forecast_year_df = forecast[forecast["ds"].dt.year == forecast_year].copy()
        forecast_year_df = add_holiday_features(forecast_year_df)

        # Prophetが学習データに含まれない定休日等を補完：全365日を確保する
        all_year_dates = pd.DataFrame({
            "ds": pd.date_range(start=f"{forecast_year}-01-01", end=f"{forecast_year}-12-31", freq="D")
        })
        forecast_year_df = all_year_dates.merge(forecast_year_df, on="ds", how="left")
        forecast_year_df["is_closed"] = forecast_year_df["is_closed"].fillna(1).astype(int)
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            forecast_year_df[col] = forecast_year_df[col].fillna(0)

        # 今年の実績データを月次補正に反映（本日以前のみ。将来入力済みデータを除外）
        actual_2026 = sales_df[
            (sales_df["ds"].dt.year == forecast_year) &
            (sales_df["y"] > 0) &
            (sales_df["ds"].dt.date < today_date)
        ].copy()
        current_month_num = date.today().month if date.today().year == forecast_year else 0

        effective_targets = {}
        for month in range(1, 13):
            month_actual = actual_2026[actual_2026["ds"].dt.month == month]
            if month < current_month_num and len(month_actual) >= 15:
                effective_targets[month] = float(month_actual["y"].mean())
                corrected_months.append(f"{month}月")
            else:
                effective_targets[month] = manual_monthly_avg.get(month, 35000) * (1 + growth_rate)

        forecast_year_df = adjust_by_manual_monthly_avg(
            forecast_year_df, effective_targets, growth_rate=0.0
        )

        # 当月（部分実績）: 実績/予測の比率で残りの日を補正
        if current_month_num > 0:
            cur_actual = actual_2026[actual_2026["ds"].dt.month == current_month_num]
            if len(cur_actual) >= 3:
                pred_on_actual = forecast_year_df[
                    forecast_year_df["ds"].isin(cur_actual["ds"].values) &
                    (forecast_year_df["is_closed"] == 0)
                ]
                if len(pred_on_actual) > 0 and pred_on_actual["yhat"].mean() > 0:
                    ratio = float(cur_actual["y"].mean()) / float(pred_on_actual["yhat"].mean())
                    remaining_mask = (
                        (forecast_year_df["ds"].dt.month == current_month_num) &
                        (~forecast_year_df["ds"].isin(cur_actual["ds"].values)) &
                        (forecast_year_df["is_closed"] == 0)
                    )
                    for col in ["yhat", "yhat_lower", "yhat_upper"]:
                        if col in forecast_year_df.columns:
                            forecast_year_df.loc[remaining_mask, col] = (
                                forecast_year_df.loc[remaining_mask, col] * ratio
                            ).clip(lower=0)
                    corrected_months.append(f"{current_month_num}月（実績比率補正）")

        # 土日祝 vs 平日の乖離補正（実績から自動計算・月次合計は維持）
        forecast_year_df, wh_factor, wd_factor = apply_weekday_weekend_correction(
            forecast_year_df, sales_df, holidays_df, forecast_year
        )
        if wh_factor is not None:
            corrected_months.append(
                f"土日祝補正 ×{wh_factor:.2f} / 平日補正 ×{wd_factor:.2f}"
            )

        if obon_multiplier > 1.0:
            forecast_year_df = apply_obon_boost(forecast_year_df, obon_multiplier=obon_multiplier)

        forecast_year_df = apply_min_daily_floor(
            forecast_year_df,
            min_monthly_avg=float(min_monthly_avg),
            min_annual_avg=float(min_annual_avg),
        )

    # ─── 予測 vs 実績の結合（比較用）───────────────────────────────
    actual_in_forecast_year = sales_df[
        (sales_df["ds"].dt.year == forecast_year) &
        (sales_df["y"] > 0) &
        (sales_df["ds"].dt.date < today_date)
    ].copy()
    has_actual = len(actual_in_forecast_year) > 0

    if has_actual:
        comparison_df = forecast_year_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(
            actual_in_forecast_year.rename(columns={"y": "actual"}),
            on="ds", how="left",
        )
        comparison_df["error"] = comparison_df["yhat"] - comparison_df["actual"]
        comparison_df["error_pct"] = (comparison_df["error"] / comparison_df["actual"] * 100).round(1)
    else:
        comparison_df = pd.DataFrame()

    # ─── キャッシュ保存 ─────────────────────────────────────────────
    st.session_state["forecast_cache"] = {
        "forecast_year_df":        forecast_year_df,
        "comparison_df":           comparison_df,
        "has_actual":              has_actual,
        "actual_in_forecast_year": actual_in_forecast_year,
        "sales_df":                sales_df,
        "historical_weather":      historical_weather,
        "forecast":                forecast,
        "holidays_df":             holidays_df,
        "closed_info":             closed_info,
        "corrected_months":        corrected_months,
    }


if corrected_months:
    st.info(f"📊 実績データを予測に反映: {', '.join(corrected_months)}")

# ─── タブ表示 ────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 日次予測", "📅 月次サマリー", "🎯 予測 vs 実績", "🔍 要因分解", "📊 過去データ"])

with tab1:
    st.subheader(f"{forecast_year}年 日次売上予測")

    # フィルター（予測実行直後は全月にリセット）
    if run_button:
        st.session_state["month_filter_select"] = "全月"
    col1, col2, col3 = st.columns(3)
    with col1:
        month_filter = st.selectbox(
            "月を選択",
            ["全月"] + [f"{m}月" for m in range(1, 13)],
            key="month_filter_select",
        )
    with col2:
        show_actual = st.checkbox("過去実績を重ねて表示", value=True)
    with col3:
        show_range = st.checkbox("信頼区間を表示", value=True)

    plot_df = forecast_year_df.copy()
    selected_month = None
    if month_filter != "全月":
        selected_month = int(month_filter.replace("月", ""))
        plot_df = plot_df[plot_df["ds"].dt.month == selected_month]

    fig = go.Figure()

    if show_actual:
        _today = date.today()
        actual_in_year = sales_df[
            (sales_df["ds"].dt.year == forecast_year) &
            (sales_df["y"] > 0) &
            (sales_df["ds"].dt.date < _today)
        ]
        if selected_month is not None:
            actual_in_year = actual_in_year[actual_in_year["ds"].dt.month == selected_month]
        if len(actual_in_year) > 0:
            fig.add_trace(go.Scatter(
                x=actual_in_year["ds"], y=actual_in_year["y"],
                mode="markers", name="実績",
                marker=dict(color="royalblue", size=4),
            ))

    if show_range:
        fig.add_trace(go.Scatter(
            x=pd.concat([plot_df["ds"], plot_df["ds"][::-1]]),
            y=pd.concat([plot_df["yhat_upper"], plot_df["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(255,127,14,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="信頼区間", hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=plot_df["ds"], y=plot_df["yhat"],
        mode="lines", name="予測",
        line=dict(color="darkorange", width=2),
    ))

    # 祝日マーカー
    holidays_in_plot = plot_df[plot_df["is_holiday"] == 1]
    if len(holidays_in_plot) > 0:
        fig.add_trace(go.Scatter(
            x=holidays_in_plot["ds"], y=holidays_in_plot["yhat"],
            mode="markers", name="祝日",
            marker=dict(color="red", size=8, symbol="star"),
        ))

    # 月フィルター選択時はx軸範囲をその月に限定
    if selected_month is not None:
        import calendar
        last_day = calendar.monthrange(forecast_year, selected_month)[1]
        xaxis_range = [
            f"{forecast_year}-{selected_month:02d}-01",
            f"{forecast_year}-{selected_month:02d}-{last_day:02d}",
        ]
        xaxis_cfg = dict(range=xaxis_range, type="date")
    else:
        xaxis_cfg = dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1ヶ月", step="month", stepmode="backward"),
                    dict(count=3, label="3ヶ月", step="month", stepmode="backward"),
                    dict(count=6, label="6ヶ月", step="month", stepmode="backward"),
                    dict(step="all", label="全期間"),
                ],
                bgcolor="rgba(240,240,240,0.8)",
                activecolor="steelblue",
            ),
            rangeslider=dict(visible=True, thickness=0.02, bgcolor="rgba(220,220,220,0.3)"),
            type="date",
        )

    fig.update_layout(
        xaxis_title="日付",
        yaxis_title="売上（円）",
        yaxis_tickformat=",",
        yaxis_range=[0, 150_000],
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=520,
        xaxis=xaxis_cfg,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 月次合計グラフ（日次グラフの直下）──
    st.markdown("**月次売上合計**")
    _open_mask_mq = (forecast_year_df["is_closed"] == 0) if "is_closed" in forecast_year_df.columns else (forecast_year_df["yhat"] > 0)
    _mq_open = forecast_year_df[_open_mask_mq]
    monthly_quick = _mq_open.groupby(_mq_open["ds"].dt.month)["yhat"].agg(
        合計="sum", 営業日平均="mean"
    ).reset_index()
    monthly_quick.columns = ["月", "合計", "営業日平均"]
    x_labels = monthly_quick["月"].astype(str) + "月"

    fig_mq = go.Figure()
    fig_mq.add_trace(go.Bar(
        x=x_labels,
        y=monthly_quick["合計"],
        name="月次合計",
        marker_color="steelblue",
        text=monthly_quick["合計"].map(lambda x: f"¥{x/10000:.0f}万"),
        textposition="outside",
        yaxis="y1",
    ))
    fig_mq.add_trace(go.Scatter(
        x=x_labels,
        y=monthly_quick["営業日平均"],
        name="営業日平均",
        mode="lines+markers",
        line=dict(color="darkorange", width=2),
        marker=dict(size=7),
        yaxis="y2",
    ))
    fig_mq.update_layout(
        yaxis=dict(
            title="月次合計（円）",
            tickformat=",",
            range=[0, monthly_quick["合計"].max() * 1.25],
        ),
        yaxis2=dict(
            title="営業日平均（円）",
            tickformat=",",
            overlaying="y",
            side="right",
            range=[0, monthly_quick["営業日平均"].max() * 1.25],
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=340,
        margin=dict(t=30, b=10),
        hovermode="x unified",
    )
    st.plotly_chart(fig_mq, use_container_width=True)

    # KPI カード（営業日ベースで集計）
    st.markdown("---")
    _open_mask_kpi = (forecast_year_df["is_closed"] == 0) if "is_closed" in forecast_year_df.columns else (forecast_year_df["yhat"] > 0)
    open_forecast = forecast_year_df[_open_mask_kpi]
    closed_forecast = forecast_year_df[~_open_mask_kpi]

    annual_avg = open_forecast["yhat"].mean() if len(open_forecast) > 0 else 0
    # 月次営業日平均の最低値
    monthly_open_avg = (
        open_forecast.groupby(open_forecast["ds"].dt.month)["yhat"].mean()
    )
    min_month_avg = monthly_open_avg.min() if len(monthly_open_avg) > 0 else 0
    min_month_label = f"{int(monthly_open_avg.idxmin())}月" if len(monthly_open_avg) > 0 else ""

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("年間予測売上合計", f"¥{open_forecast['yhat'].sum():,.0f}",
                  help=f"営業日{len(open_forecast)}日・休業日{len(closed_forecast)}日")
    with k2:
        floor_ok = "✅" if annual_avg >= min_annual_avg else "⚠️"
        st.metric("年間 営業日平均", f"¥{annual_avg:,.0f}",
                  delta=f"{floor_ok} 下限¥{min_annual_avg:,}",
                  delta_color="off")
    with k3:
        floor_ok_m = "✅" if min_month_avg >= min_monthly_avg else "⚠️"
        st.metric(f"最低月平均（{min_month_label}）", f"¥{min_month_avg:,.0f}",
                  delta=f"{floor_ok_m} 下限¥{min_monthly_avg:,}",
                  delta_color="off")
    with k4:
        if len(open_forecast) > 0:
            best_day = open_forecast.loc[open_forecast["yhat"].idxmax()]
            st.metric("最高予測日", f"{best_day['ds'].strftime('%m/%d')} ¥{best_day['yhat']:,.0f}")

    # 詳細テーブル
    with st.expander("📋 日次データ一覧"):
        dow_map = {"Monday": "月", "Tuesday": "火", "Wednesday": "水", "Thursday": "木",
                   "Friday": "金", "Saturday": "土", "Sunday": "日"}
        _cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
        if "is_closed" in forecast_year_df.columns:
            _cols.append("is_closed")
        display_df = forecast_year_df[_cols].copy()
        display_df["曜日"] = display_df["ds"].dt.day_name().map(dow_map)
        display_df["日付"] = display_df["ds"].dt.strftime("%Y-%m-%d")
        is_closed_col = display_df["is_closed"] if "is_closed" in display_df.columns else (display_df["yhat"] == 0).astype(int)
        display_df["予測売上"] = display_df.apply(
            lambda r: "休業日" if is_closed_col[r.name] == 1 else f"¥{r['yhat']:,.0f}", axis=1
        )
        display_df["下限"] = display_df.apply(
            lambda r: "―" if is_closed_col[r.name] == 1 else f"¥{r['yhat_lower']:,.0f}", axis=1
        )
        display_df["上限"] = display_df.apply(
            lambda r: "―" if is_closed_col[r.name] == 1 else f"¥{r['yhat_upper']:,.0f}", axis=1
        )
        def _morning_rice(lo: int) -> int:
            if lo < 13:
                return 8
            elif lo <= 16:
                return 10
            elif lo <= 18:
                return 12
            else:
                return 16

        def _rice_cols(r):
            closed = is_closed_col[r.name] == 1
            if closed:
                return pd.Series({"推定炊飯量": "―", "朝イチの炊飯量": "―", "追加の炊飯量": "―"})
            lo = round((r["yhat"] + r["yhat_lower"]) / 2 / 2000)
            hi = round((r["yhat_lower"] + r["yhat_upper"]) / 2 / 2000)
            est = f"{lo}合" if lo >= hi else f"{lo}合〜{hi}合"
            morning = _morning_rice(lo)
            add_lo = max(lo - morning, 0)
            add_hi = max(hi - morning, 0)
            if add_lo == 0 and add_hi == 0:
                add_str = "0合"
            elif add_lo >= add_hi:
                add_str = f"{add_lo}合"
            else:
                add_str = f"{add_lo}合〜{add_hi}合"
            return pd.Series({"推定炊飯量": est, "朝イチの炊飯量": f"{morning}合", "追加の炊飯量": add_str})

        display_df[["推定炊飯量", "朝イチの炊飯量", "追加の炊飯量"]] = display_df.apply(_rice_cols, axis=1)
        display_df = display_df[["日付", "曜日", "予測売上", "下限", "上限", "推定炊飯量", "朝イチの炊飯量", "追加の炊飯量"]]
        st.dataframe(display_df.set_index("日付"), use_container_width=True)


with tab2:
    st.subheader(f"{forecast_year}年 月次サマリー")

    # 休業日を除いた月次集計
    _open_mask_tab2 = (forecast_year_df["is_closed"] == 0) if "is_closed" in forecast_year_df.columns else (forecast_year_df["yhat"] > 0)
    open_year_df = forecast_year_df[_open_mask_tab2]
    monthly = forecast_year_df.groupby(forecast_year_df["ds"].dt.month).agg(
        予測売上合計=("yhat", "sum"),
        祝日数=("is_holiday", "sum"),
        休業日数=("is_closed", "sum") if "is_closed" in forecast_year_df.columns else ("yhat", lambda x: (x == 0).sum()),
    ).reset_index()
    monthly.columns = ["月", "予測売上合計", "祝日数", "休業日数"]
    # 営業日平均 = 合計 ÷ (日数 - 休業日数)
    days_in_month = forecast_year_df.groupby(forecast_year_df["ds"].dt.month).size().values
    monthly["営業日数"] = days_in_month - monthly["休業日数"].values
    monthly["営業日平均売上"] = (monthly["予測売上合計"] / monthly["営業日数"].replace(0, np.nan)).fillna(0)
    monthly["月"] = monthly["月"].astype(int)

    month_labels = monthly["月"].astype(str) + "月"

    # ── グラフ①：月次売上合計 ──
    st.markdown("**月次売上合計**")
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(
        x=month_labels,
        y=monthly["予測売上合計"],
        text=monthly["予測売上合計"].map(lambda x: f"¥{x/10000:.0f}万"),
        textposition="outside",
        marker_color="steelblue",
        name="予測売上合計",
    ))
    fig_monthly.update_layout(
        xaxis_title="月", yaxis_title="売上合計（円）",
        yaxis_tickformat=",",
        yaxis_range=[0, monthly["予測売上合計"].max() * 1.15],
        height=350, margin=dict(t=20),
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    # 月次営業日平均をフロア適用後の値で上書き
    open_avg_by_month = (
        open_forecast.groupby(open_forecast["ds"].dt.month)["yhat"].mean()
    )
    monthly["営業日平均売上"] = monthly["月"].map(open_avg_by_month).fillna(0)
    monthly["月次下限達成"] = monthly["営業日平均売上"].apply(
        lambda x: "✅" if x >= min_monthly_avg else "⚠️ 下限未満"
    )

    # ── グラフ②：営業日平均売上（下限ライン付き）──
    st.markdown("**営業日あたり平均売上**")
    bar_colors = [
        "steelblue" if v >= min_monthly_avg else "tomato"
        for v in monthly["営業日平均売上"]
    ]
    fig_avg = go.Figure()
    fig_avg.add_trace(go.Bar(
        x=month_labels,
        y=monthly["営業日平均売上"],
        text=monthly["営業日平均売上"].map(lambda x: f"¥{x:,.0f}"),
        textposition="outside",
        marker_color=bar_colors,
        name="営業日平均",
    ))
    # 月次下限ライン
    fig_avg.add_hline(
        y=min_monthly_avg,
        line_dash="dash", line_color="tomato",
        annotation_text=f"月次下限 ¥{min_monthly_avg:,}",
        annotation_position="top left",
    )
    # 年間下限ライン
    fig_avg.add_hline(
        y=min_annual_avg,
        line_dash="dot", line_color="orange",
        annotation_text=f"年間下限 ¥{min_annual_avg:,}",
        annotation_position="top right",
    )
    fig_avg.update_layout(
        xaxis_title="月", yaxis_title="営業日平均（円）",
        yaxis_tickformat=",",
        yaxis_range=[0, monthly["営業日平均売上"].max() * 1.2],
        height=380, margin=dict(t=20),
    )
    st.plotly_chart(fig_avg, use_container_width=True)

    # ── テーブル ──
    monthly_display = monthly.copy()
    monthly_display["予測売上合計"] = monthly_display["予測売上合計"].map(lambda x: f"¥{x:,.0f}")
    monthly_display["営業日平均売上"] = monthly_display["営業日平均売上"].map(lambda x: f"¥{x:,.0f}")
    monthly_display["祝日数"] = monthly_display["祝日数"].astype(int)
    monthly_display["休業日数"] = monthly_display["休業日数"].astype(int)
    monthly_display["営業日数"] = monthly_display["営業日数"].astype(int)
    st.dataframe(monthly_display.set_index("月"), use_container_width=True)

with tab3:
    st.subheader("🎯 予測 vs 実績 比較")

    if not has_actual:
        st.info(f"{forecast_year}年の実績データがまだありません。実績が入力されると自動で比較が表示されます。")
    else:
        # 月次比較
        monthly_cmp = comparison_df.dropna(subset=["actual"]).groupby(
            comparison_df["ds"].dt.month
        ).agg(
            予測=("yhat", "sum"),
            実績=("actual", "sum"),
        ).reset_index()
        monthly_cmp.columns = ["月", "予測", "実績"]
        monthly_cmp["達成率"] = (monthly_cmp["実績"] / monthly_cmp["予測"] * 100).round(1)
        monthly_cmp["差分"] = monthly_cmp["実績"] - monthly_cmp["予測"]

        # 月次棒グラフ
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            name="予測", x=monthly_cmp["月"].astype(str) + "月",
            y=monthly_cmp["予測"], marker_color="lightsalmon",
        ))
        fig_cmp.add_trace(go.Bar(
            name="実績", x=monthly_cmp["月"].astype(str) + "月",
            y=monthly_cmp["実績"], marker_color="steelblue",
        ))
        fig_cmp.update_layout(
            barmode="group", yaxis_tickformat=",", height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # 達成率ゲージ
        cols = st.columns(len(monthly_cmp))
        for i, row in monthly_cmp.iterrows():
            rate = row["達成率"]
            delta_color = "normal" if rate >= 95 else "inverse"
            cols[i].metric(
                label=f"{int(row['月'])}月",
                value=f"{rate:.1f}%",
                delta=f"¥{row['差分']:+,.0f}",
                delta_color=delta_color,
            )

        st.markdown("---")

        # 日次比較グラフ
        st.markdown("**日次 予測 vs 実績**")
        daily_cmp = comparison_df.dropna(subset=["actual"]).copy()
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(
            x=daily_cmp["ds"], y=daily_cmp["actual"],
            mode="markers", name="実績",
            marker=dict(color="steelblue", size=5),
        ))
        fig_daily.add_trace(go.Scatter(
            x=daily_cmp["ds"], y=daily_cmp["yhat"],
            mode="lines", name="予測",
            line=dict(color="darkorange", width=2),
        ))
        # 誤差が大きい日（±20%超）をハイライト
        big_error = daily_cmp[daily_cmp["error_pct"].abs() > 20]
        if len(big_error) > 0:
            fig_daily.add_trace(go.Scatter(
                x=big_error["ds"], y=big_error["actual"],
                mode="markers", name="誤差±20%超",
                marker=dict(color="red", size=9, symbol="circle-open", line=dict(width=2)),
            ))
        fig_daily.update_layout(
            yaxis_tickformat=",", height=380, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_daily, use_container_width=True)

        # 月次サマリーテーブル
        with st.expander("📋 月次比較テーブル"):
            tbl = monthly_cmp.copy()
            tbl["予測"] = tbl["予測"].map(lambda x: f"¥{x:,.0f}")
            tbl["実績"] = tbl["実績"].map(lambda x: f"¥{x:,.0f}")
            tbl["差分"] = tbl["差分"].map(lambda x: f"¥{x:+,.0f}")
            tbl["達成率"] = tbl["達成率"].map(lambda x: f"{x:.1f}%")
            st.dataframe(tbl.set_index("月"), use_container_width=True)

with tab4:
    st.subheader("売上予測の要因分解")
    st.caption("各要因が売上にどれだけ影響しているか（乗算モデルのため相対値）")

    # Prophet コンポーネントプロット
    component_cols = [c for c in ["trend", "weekly", "yearly", "monthly", "holidays"] if c in forecast.columns]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📈 トレンド**")
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=forecast["ds"], y=forecast["trend"], mode="lines", line=dict(color="steelblue")))
        fig_trend.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), yaxis_tickformat=",")
        st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        st.markdown("**📅 週次季節性（曜日効果）**")
        if "weekly" in forecast.columns:
            weekly_avg = forecast.groupby(forecast["ds"].dt.dayofweek)["weekly"].mean().reset_index()
            weekly_avg.columns = ["dayofweek", "weekly"]
            dow_labels = ["月", "火", "水", "木", "金", "土", "日"]
            weekly_avg["曜日"] = weekly_avg["dayofweek"].map(lambda x: dow_labels[x])
            fig_weekly = go.Figure(go.Bar(
                x=weekly_avg["曜日"], y=weekly_avg["weekly"],
                marker_color=["steelblue"] * 5 + ["tomato"] * 2,
            ))
            fig_weekly.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_weekly, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**🗓️ 年次季節性（月次変動）**")
        if "yearly" in forecast.columns:
            yearly_avg = forecast.groupby(forecast["ds"].dt.month)["yearly"].mean().reset_index()
            yearly_avg.columns = ["month", "yearly"]
            yearly_avg["月"] = yearly_avg["month"].astype(str) + "月"
            fig_yearly = go.Figure(go.Bar(x=yearly_avg["月"], y=yearly_avg["yearly"], marker_color="mediumseagreen"))
            fig_yearly.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_yearly, use_container_width=True)

    with col4:
        st.markdown("**🎌 祝日効果**")
        if "holidays" in forecast.columns:
            holiday_effect = forecast[forecast["holidays"] != 0][["ds", "holidays"]].copy()
            if len(holiday_effect) > 0:
                holiday_effect = holiday_effect.merge(
                    holidays_df[["ds", "holiday"]].rename(columns={"holiday": "祝日名"}),
                    on="ds", how="left",
                )
                holiday_effect = holiday_effect.groupby("祝日名")["holidays"].mean().sort_values()
                fig_hol = go.Figure(go.Bar(
                    x=holiday_effect.values,
                    y=holiday_effect.index,
                    orientation="h",
                    marker_color=["tomato" if v < 0 else "steelblue" for v in holiday_effect.values],
                ))
                fig_hol.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_hol, use_container_width=True)

with tab5:
    st.subheader("過去データ概要")

    # 休業日パターン表示
    with st.expander("🗓️ 休業日パターン分析", expanded=True):
        st.dataframe(closed_info["stats"].set_index("曜日"), use_container_width=True)
        if closed_info["closed_weekdays"]:
            dow_labels_map = {0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"}
            names = "・".join(dow_labels_map[d] for d in sorted(closed_info["closed_weekdays"]))
            st.write(f"**定休日（休業率70%以上）：{names}曜日**")
        if closed_info["closed_dates"]:
            st.write(f"臨時休業日（定休日以外で売上0円）：{len(closed_info['closed_dates'])}日")

    fig_hist = go.Figure()
    # 営業日と休業日を色分けして表示
    open_df = sales_df[sales_df["y"] > 0]
    closed_df = sales_df[sales_df["y"] == 0]
    if len(closed_df) > 0:
        fig_hist.add_trace(go.Scatter(
            x=closed_df["ds"], y=closed_df["y"],
            mode="markers", name="休業日",
            marker=dict(color="lightgray", size=4),
        ))
    fig_hist.add_trace(go.Scatter(
        x=open_df["ds"], y=open_df["y"],
        mode="lines", name="実績売上（営業日）",
        line=dict(color="steelblue", width=1),
    ))
    # 30日移動平均
    sales_df_sorted = sales_df.sort_values("ds")
    sales_df_sorted["ma30"] = sales_df_sorted["y"].rolling(30, center=True).mean()
    fig_hist.add_trace(go.Scatter(
        x=sales_df_sorted["ds"], y=sales_df_sorted["ma30"],
        mode="lines", name="30日移動平均",
        line=dict(color="darkorange", width=2),
    ))
    fig_hist.update_layout(
        xaxis_title="日付", yaxis_title="売上（円）",
        yaxis_tickformat=",", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # 曜日別平均
    st.markdown("**曜日別 平均売上**")
    dow_labels = ["月", "火", "水", "木", "金", "土", "日"]
    dow_avg = sales_df.copy()
    dow_avg["曜日"] = dow_avg["ds"].dt.dayofweek.map(lambda x: dow_labels[x])
    dow_avg = dow_avg.groupby("曜日")["y"].mean().reindex(dow_labels)
    fig_dow = go.Figure(go.Bar(
        x=dow_avg.index, y=dow_avg.values,
        marker_color=["steelblue"] * 5 + ["tomato"] * 2,
        text=dow_avg.values.astype(int),
        texttemplate="¥%{text:,}",
        textposition="outside",
    ))
    fig_dow.update_layout(yaxis_tickformat=",", height=300)
    st.plotly_chart(fig_dow, use_container_width=True)

    # 天候データが揃っている場合
    if len(historical_weather) > 0 and "temp_avg" in historical_weather.columns:
        weather_sales = sales_df.merge(
            historical_weather[["ds", "temp_avg", "is_rain"]],
            on="ds", how="inner",
        )
        if len(weather_sales) > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**気温と売上の関係**")
                fig_temp = px.scatter(
                    weather_sales, x="temp_avg", y="y",
                    trendline="ols", opacity=0.4,
                    labels={"temp_avg": "平均気温(℃)", "y": "売上(円)"},
                )
                fig_temp.update_layout(height=300, yaxis_tickformat=",")
                st.plotly_chart(fig_temp, use_container_width=True)

            with col2:
                st.markdown("**雨天の影響（月別・季節補正済み）**")
                # 単純平均では季節の混同が生じるため、同月内で比較する
                ws = weather_sales[weather_sales["y"] > 0].copy()
                ws["month"] = ws["ds"].dt.month
                rain_by_month = (
                    ws.groupby(["month", "is_rain"])["y"]
                    .mean().unstack(fill_value=np.nan)
                )
                if 0 in rain_by_month.columns and 1 in rain_by_month.columns:
                    rain_by_month["effect"] = rain_by_month[1] - rain_by_month[0]
                    rain_by_month = rain_by_month.dropna(subset=["effect"])
                    colors = ["tomato" if v < 0 else "steelblue" for v in rain_by_month["effect"]]
                    fig_rain = go.Figure(go.Bar(
                        x=[f"{m}月" for m in rain_by_month.index],
                        y=rain_by_month["effect"],
                        marker_color=colors,
                        text=rain_by_month["effect"].map(lambda x: f"¥{x:+,.0f}"),
                        textposition="outside",
                    ))
                    fig_rain.add_hline(y=0, line_color="gray", line_width=1)
                    avg_effect = rain_by_month["effect"].mean()
                    fig_rain.update_layout(
                        height=300, yaxis_tickformat=",",
                        xaxis_title="月", yaxis_title="雨天日 − 晴天日（円）",
                    )
                    st.plotly_chart(fig_rain, use_container_width=True)
                    direction = f"▼ 平均 ¥{abs(avg_effect):,.0f} 減" if avg_effect < 0 else f"▲ 平均 ¥{avg_effect:,.0f} 増"
                    st.caption(f"月内比較による雨天影響: {direction}（赤＝雨で減収、青＝雨で増収）")
