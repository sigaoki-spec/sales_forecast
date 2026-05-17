"""Google Trends から館山の検索量を取得してリグレッサーに使う"""
import time
import pandas as pd
import numpy as np
from datetime import date, timedelta

KEYWORDS = ["館山", "館山 海水浴"]


def fetch_google_trends(
    keywords: list = None,
    start_date: date = None,
    end_date: date = None,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Google Trends から検索量を取得し、日次に補間して返す。
    長期間は週次データになるため線形補間で日次化する。
    戻り値: ds, trends_index (0〜100 に正規化)
    """
    from pytrends.request import TrendReq

    if keywords is None:
        keywords = KEYWORDS
    if start_date is None:
        start_date = date.today() - timedelta(days=365 * 3)
    if end_date is None:
        end_date = date.today()

    timeframe = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"

    for attempt in range(max_retries):
        try:
            pytrends = TrendReq(hl="ja-JP", tz=540, timeout=(10, 30))
            pytrends.build_payload(keywords, timeframe=timeframe, geo="JP")
            raw = pytrends.interest_over_time()
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(10 * (attempt + 1))
            else:
                raise RuntimeError(f"Google Trends 取得失敗: {e}")

    if raw.empty:
        return pd.DataFrame(columns=["ds", "trends_index"])

    raw = raw.drop(columns=["isPartial"], errors="ignore")

    # キーワードごとの平均を合成指数にする
    raw["trends_index"] = raw[keywords].mean(axis=1)
    raw = raw[["trends_index"]].reset_index()
    raw = raw.rename(columns={"date": "ds"})
    raw["ds"] = pd.to_datetime(raw["ds"])

    # 週次 → 日次に線形補間
    all_days = pd.date_range(start=raw["ds"].min(), end=raw["ds"].max(), freq="D")
    df = raw.set_index("ds").reindex(all_days).interpolate(method="linear").reset_index()
    df.columns = ["ds", "trends_index"]

    # 0〜100 に正規化
    max_val = df["trends_index"].max()
    if max_val > 0:
        df["trends_index"] = (df["trends_index"] / max_val * 100).round(2)

    return df


def estimate_future_trends(
    historical_trends: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    過去の季節パターン（月・日の平均）から将来のトレンド指数を推定する。
    """
    if historical_trends.empty or len(historical_trends) < 30:
        return pd.DataFrame({
            "ds": future_dates,
            "trends_index": 50.0,
        })

    hist = historical_trends.copy()
    hist["month"] = hist["ds"].dt.month
    hist["day"] = hist["ds"].dt.day

    seasonal = (
        hist.groupby(["month", "day"])["trends_index"]
        .mean()
        .reset_index()
    )

    future_df = pd.DataFrame({"ds": future_dates})
    future_df["month"] = future_df["ds"].dt.month
    future_df["day"] = future_df["ds"].dt.day
    future_df = future_df.merge(seasonal, on=["month", "day"], how="left")

    global_mean = hist["trends_index"].mean()
    future_df["trends_index"] = future_df["trends_index"].fillna(global_mean)

    return future_df[["ds", "trends_index"]]
