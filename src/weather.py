"""Open-Meteo API から過去の天候・気温データを取得する"""
import requests
import pandas as pd
from datetime import date, timedelta


WEATHER_CODES = {
    0: "快晴", 1: "晴れ", 2: "一部曇り", 3: "曇り",
    45: "霧", 48: "霧",
    51: "霧雨(弱)", 53: "霧雨", 55: "霧雨(強)",
    61: "小雨", 63: "雨", 65: "大雨",
    71: "小雪", 73: "雪", 75: "大雪",
    77: "霰", 80: "にわか雨(弱)", 81: "にわか雨", 82: "にわか雨(強)",
    85: "にわか雪", 86: "にわか雪(強)",
    95: "雷雨", 96: "雷雨(霰)", 99: "雷雨(大粒霰)",
}


def _is_rain(code: int) -> int:
    return int(code in {51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 96, 99})


def _is_snow(code: int) -> int:
    return int(code in {71, 73, 75, 77, 85, 86})


def fetch_historical_weather(
    lat: float, lon: float, start: date, end: date
) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "Asia/Tokyo",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["daily"]

    df = pd.DataFrame({
        "ds": pd.to_datetime(data["time"]),
        "weather_code": data["weathercode"],
        "temp_max": data["temperature_2m_max"],
        "temp_min": data["temperature_2m_min"],
        "precipitation": data["precipitation_sum"],
    })
    df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
    df["is_rain"] = df["weather_code"].apply(_is_rain)
    df["is_snow"] = df["weather_code"].apply(_is_snow)
    df["weather_label"] = df["weather_code"].map(WEATHER_CODES).fillna("不明")
    return df


def fetch_forecast_weather(lat: float, lon: float, days: int = 16) -> pd.DataFrame:
    """直近 16 日間の予報を取得する"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "Asia/Tokyo",
        "forecast_days": days,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["daily"]

    df = pd.DataFrame({
        "ds": pd.to_datetime(data["time"]),
        "weather_code": data["weathercode"],
        "temp_max": data["temperature_2m_max"],
        "temp_min": data["temperature_2m_min"],
        "precipitation": data["precipitation_sum"],
    })
    df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
    df["is_rain"] = df["weather_code"].apply(_is_rain)
    df["is_snow"] = df["weather_code"].apply(_is_snow)
    df["weather_label"] = df["weather_code"].map(WEATHER_CODES).fillna("不明")
    return df


def estimate_future_weather_from_history(
    historical_df: pd.DataFrame, future_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """過去同日の平均値を将来の天候推定に使う（予報期間外）"""
    hist = historical_df.copy()
    hist["month_day"] = hist["ds"].dt.strftime("%m-%d")

    avg = hist.groupby("month_day").agg(
        temp_avg=("temp_avg", "mean"),
        precipitation=("precipitation", "mean"),
        is_rain=("is_rain", "mean"),
        is_snow=("is_snow", "mean"),
    ).reset_index()

    future_df = pd.DataFrame({"ds": future_dates})
    future_df["month_day"] = future_df["ds"].dt.strftime("%m-%d")
    future_df = future_df.merge(avg, on="month_day", how="left")
    future_df["weather_code"] = 0
    future_df["weather_label"] = "（過去平均推定）"
    future_df = future_df.drop(columns=["month_day"])
    return future_df
