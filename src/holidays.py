"""日本の祝日・特別日フラグを付与する"""
from __future__ import annotations

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
    # 若潮マラソン（館山）：開催日は来店増。営業7:45-17:00と長時間営業
    (2026, 1, 25, 0, 0, "若潮マラソン"),
    (2027, 1, 31, 0, 0, "若潮マラソン"),
]

# 地方の休日（jpholiday 非対応）：毎年同じ月日・振替休日なし（月, 日, 名称）
PREFECTURAL_HOLIDAYS = [
    (6, 15, "千葉県民の日"),  # 千葉県は学校休業。館山店の来店に影響
]


def _prefectural_holiday_name(d: datetime.date) -> str:
    """県民の日など地方の休日名を返す（該当なしは空文字）。"""
    for month, day, name in PREFECTURAL_HOLIDAYS:
        if d.month == month and d.day == day:
            return name
    return ""


def _holiday_name_for(d: datetime.date) -> str:
    """国民の祝日＋地方の休日を合わせた休日名（該当なしは空文字）。"""
    return jpholiday.is_holiday_name(d) or _prefectural_holiday_name(d)


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["holiday_name"] = df["ds"].apply(lambda d: _holiday_name_for(d.date()))
    df["is_holiday"] = (df["holiday_name"] != "").astype(int)
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


def build_prophet_holidays(through_year: int | None = None) -> pd.DataFrame:
    """
    Prophet の holidays DataFrame を構築する。
    公式祝日に加え、お盆・GW・年末年始を特別期間として追加する。

    through_year: 祝日を生成する最終年（予測対象年）。未指定なら翌々年まで。
    """
    current_year = datetime.date.today().year
    # 予測対象年が将来でも必ずカバーするよう上限を決める（過去は学習用に4年分）
    last_year = max(current_year + 2, (through_year or 0) + 1)
    rows = []

    for year in range(current_year - 4, last_year + 1):
        # ① 公式祝日
        start = datetime.date(year, 1, 1)
        end = datetime.date(year, 12, 31)
        d = start
        while d <= end:
            name = _holiday_name_for(d)
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
