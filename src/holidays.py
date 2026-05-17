"""日本の祝日・特別日フラグを付与する"""
import datetime
import jpholiday
import pandas as pd


# お盆ウィークの定義（8/10〜8/18、中心は8/13〜16）
# lower_window: 中心日より前に何日広げるか（負の値）
# upper_window: 中心日より後に何日広げるか
OBON_CENTER = (8, 15)  # 8月15日を中心とする
OBON_LOWER = -5        # 8/10から
OBON_UPPER = 3         # 8/18まで

# GWの定義（5/3〜5/6が祝日だが、前後も含めて5/1〜5/6を特別期間に）
GW_CENTER = (5, 4)
GW_LOWER = -3
GW_UPPER = 2

# 年末年始（12/29〜1/3）
NEWYEAR_EVE_CENTER = (12, 31)
NEWYEAR_EVE_LOWER = -2
NEWYEAR_EVE_UPPER = 0

# 毎年同じ日付の地域イベント（月, 日, lower, upper, イベント名）
LOCAL_EVENTS = [
    (8, 8, 1, 1, "花火大会"),  # 8月8日、前日1日・翌日1日も効果あり
]

# 年によって日付が変わるイベント（年, 月, 日, lower, upper, イベント名）
LOCAL_EVENTS_BY_YEAR = [
    (2025, 9, 14, 1, 1, "秋祭り"),
    (2026, 9, 20, 1, 1, "秋祭り"),
]


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_holiday"] = df["ds"].apply(lambda d: int(jpholiday.is_holiday(d.date())))
    df["holiday_name"] = df["ds"].apply(
        lambda d: jpholiday.is_holiday_name(d.date()) or ""
    )
    df["day_of_week"] = df["ds"].dt.dayofweek  # 0=月, 6=日
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)

    # 祝日前日・翌日フラグ
    holiday_dates = set(df.loc[df["is_holiday"] == 1, "ds"].dt.date)
    df["is_pre_holiday"] = df["ds"].apply(
        lambda d: int((d + pd.Timedelta(days=1)).date() in holiday_dates)
    )
    df["is_post_holiday"] = df["ds"].apply(
        lambda d: int((d - pd.Timedelta(days=1)).date() in holiday_dates)
    )

    # お盆フラグ（8/10〜8/18）
    df["is_obon"] = df["ds"].apply(
        lambda d: int(d.month == 8 and 10 <= d.day <= 18)
    )
    return df


def _make_special_period_rows(year: int, month: int, day: int,
                               lower: int, upper: int, name: str) -> list:
    """特別期間の中心日から lower〜upper 日を1行ずつ生成する"""
    rows = []
    center = datetime.date(year, month, day)
    for offset in range(lower, upper + 1):
        d = center + datetime.timedelta(days=offset)
        if d.year != year and name != "年末年始":
            continue
        rows.append({
            "holiday": name,
            "ds": pd.Timestamp(d),
            "lower_window": 0,
            "upper_window": 0,
        })
    return rows


def build_prophet_holidays() -> pd.DataFrame:
    """
    Prophet の holidays DataFrame を構築する。
    公式祝日に加え、お盆・GW・年末年始を特別期間として追加する。
    """
    current_year = datetime.date.today().year
    rows = []

    for year in range(current_year - 4, current_year + 2):
        # ① 公式祝日
        start = datetime.date(year, 1, 1)
        end = datetime.date(year, 12, 31)
        d = start
        while d <= end:
            name = jpholiday.is_holiday_name(d)
            if name:
                rows.append({
                    "holiday": name,
                    "ds": pd.Timestamp(d),
                    "lower_window": 0,
                    "upper_window": 0,
                })
            d += datetime.timedelta(days=1)

        # ② お盆ウィーク（8/10〜8/18）
        rows += _make_special_period_rows(
            year, *OBON_CENTER, OBON_LOWER, OBON_UPPER, "お盆ウィーク"
        )

        # ③ GW前後（5/1〜5/6）
        rows += _make_special_period_rows(
            year, *GW_CENTER, GW_LOWER, GW_UPPER, "GW特別期間"
        )

        # ④ 年末（12/29〜12/31）
        rows += _make_special_period_rows(
            year, *NEWYEAR_EVE_CENTER, NEWYEAR_EVE_LOWER, NEWYEAR_EVE_UPPER, "年末"
        )

        # ⑤ 毎年同じ日付の地域イベント（花火大会など）
        for month, day, lower, upper, name in LOCAL_EVENTS:
            try:
                d = datetime.date(year, month, day)
                rows.append({
                    "holiday": name,
                    "ds": pd.Timestamp(d),
                    "lower_window": -lower,
                    "upper_window": upper,
                })
            except ValueError:
                continue

    # ⑥ 年によって日付が変わるイベント（祭りなど）
    for yr, month, day, lower, upper, name in LOCAL_EVENTS_BY_YEAR:
        try:
            d = datetime.date(yr, month, day)
            rows.append({
                "holiday": name,
                "ds": pd.Timestamp(d),
                "lower_window": -lower,
                "upper_window": upper,
            })
        except ValueError:
            continue

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["holiday", "ds", "lower_window", "upper_window"]
    )
    # 重複除去（公式祝日とお盆等が同日の場合）
    df = df.drop_duplicates(subset=["ds"]).reset_index(drop=True)
    return df
