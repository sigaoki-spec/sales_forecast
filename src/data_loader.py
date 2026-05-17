"""Google Sheets から売上データを読み込む"""
import os
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

load_dotenv()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def load_from_google_sheets(
    spreadsheet_id: str,
    credentials_path: str,
    sheet_name: str = None,
    date_col: int = 0,
    sales_col: int = 1,
    header_row: int = 1,
) -> pd.DataFrame:
    creds = Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    client = gspread.authorize(creds)

    spreadsheet = client.open_by_key(spreadsheet_id)
    worksheet = spreadsheet.get_worksheet(0) if sheet_name is None else spreadsheet.worksheet(sheet_name)

    all_values = worksheet.get_all_values()
    if header_row > 0:
        all_values = all_values[header_row:]

    rows = []
    for row in all_values:
        if len(row) <= max(date_col, sales_col):
            continue
        date_str = row[date_col].strip()
        sales_str = row[sales_col].strip().replace(",", "").replace("¥", "").replace("￥", "")
        if not date_str or not sales_str:
            continue
        try:
            rows.append({"date_str": date_str, "sales_str": sales_str})
        except Exception:
            continue

    df = pd.DataFrame(rows)
    df["ds"] = pd.to_datetime(df["date_str"], infer_datetime_format=True)
    df["y"] = pd.to_numeric(df["sales_str"], errors="coerce")
    df = df.dropna(subset=["ds", "y"])
    df = df[["ds", "y"]].sort_values("ds").reset_index(drop=True)
    return df


def load_from_google_sheets_horizontal(
    spreadsheet_id: str,
    credentials_path: str,
    sheet_name: str = None,
    date_row: int = 2,
    sales_row: int = 29,
    start_col: int = 1,
) -> pd.DataFrame:
    """横向きレイアウト用：日付と売上がそれぞれ1行に並んでいる場合"""
    creds = Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    client = gspread.authorize(creds)

    spreadsheet = client.open_by_key(spreadsheet_id)
    worksheet = spreadsheet.get_worksheet(0) if sheet_name is None else spreadsheet.worksheet(sheet_name)

    all_values = worksheet.get_all_values()

    # 行番号は1始まりで指定されるので0始まりに変換
    date_row_idx = date_row - 1
    sales_row_idx = sales_row - 1

    if len(all_values) <= max(date_row_idx, sales_row_idx):
        raise ValueError(f"シートの行数が足りません（指定行: 日付={date_row}, 売上={sales_row}）")

    date_values = all_values[date_row_idx][start_col:]
    sales_values = all_values[sales_row_idx][start_col:]

    rows = []
    skipped = []
    for i, (date_str, sales_str) in enumerate(zip(date_values, sales_values)):
        date_str = date_str.strip()
        sales_str = (sales_str.strip()
                     .replace(",", "")
                     .replace("¥", "")
                     .replace("￥", "")
                     .replace(" ", ""))
        # 日付・売上どちらかが空のセルはスキップ
        if not date_str or not sales_str:
            continue
        # 日付として明らかに無効な値（曜日・ラベル等）はスキップ
        if date_str in ("土", "日", "月", "火", "水", "木", "金"):
            continue
        try:
            parsed_date = pd.to_datetime(date_str)
            parsed_sales = float(sales_str)
            rows.append({"date_str": date_str, "ds": parsed_date, "y": parsed_sales})
        except Exception as e:
            skipped.append(f"列{start_col + i + 1}: 日付='{date_str}' 売上='{sales_str}' → {e}")
            continue

    if skipped:
        import warnings
        warnings.warn(f"読み飛ばしたセル {len(skipped)}件: {skipped[:3]}")

    if not rows:
        raise ValueError(
            "有効なデータが1件も読み込めませんでした。\n"
            f"・日付行({date_row}行目)の先頭5セル: {list(date_values[:5])}\n"
            f"・売上行({sales_row}行目)の先頭5セル: {list(sales_values[:5])}\n"
            "「データ開始列番号」が正しいか確認してください。"
        )

    df = pd.DataFrame(rows)[["ds", "y"]].sort_values("ds").reset_index(drop=True)
    return df


def _col_to_letter(col: int) -> str:
    """列番号（1始まり）をA1表記のアルファベットに変換する"""
    result = ""
    while col > 0:
        col, remainder = divmod(col - 1, 26)
        result = chr(65 + remainder) + result
    return result


def write_forecast_to_sheets(
    spreadsheet_id: str,
    credentials_path: str,
    forecast_df: pd.DataFrame,
    sheet_name: str = None,
    date_row: int = 2,
    target_row: int = 23,
    start_col: int = 1,
) -> dict:
    """
    予測結果をスプレッドシートの指定行に書き込む。
    date_row 行の日付と forecast_df の ds を照合し、
    一致する列の target_row 行に yhat（整数）を書き込む。
    戻り値: {"written": 書き込み件数, "matched": マッチ件数, "date_sample": 日付サンプル}
    """
    creds = Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    client = gspread.authorize(creds)

    spreadsheet = client.open_by_key(spreadsheet_id)
    worksheet = (
        spreadsheet.get_worksheet(0) if sheet_name is None
        else spreadsheet.worksheet(sheet_name)
    )

    all_values = worksheet.get_all_values()
    if date_row > len(all_values):
        raise ValueError(f"日付行（{date_row}行目）がシートの行数を超えています。")
    date_row_values = all_values[date_row - 1]

    # 予測データを {date: (yhat, yhat_lower, yhat_upper)} の辞書に変換
    forecast_dict = {}
    for _, row in forecast_df.iterrows():
        d = row["ds"].date()
        if row.get("is_closed", 0) == 1:
            forecast_dict[d] = (0, 0, 0)
        else:
            forecast_dict[d] = (
                int(round(max(row["yhat"], 0))),
                int(round(max(row.get("yhat_lower", row["yhat"]), 0))),
                int(round(max(row.get("yhat_upper", row["yhat"]), 0))),
            )

    updates = []
    date_sample = []
    for col_idx, date_str in enumerate(date_row_values):
        if col_idx < start_col:
            continue
        date_str = date_str.strip()
        if not date_str:
            continue
        try:
            parsed_date = pd.to_datetime(date_str).date()
        except Exception:
            continue
        if len(date_sample) < 3:
            date_sample.append(str(parsed_date))
        if parsed_date in forecast_dict:
            yhat, yhat_lower, yhat_upper = forecast_dict[parsed_date]
            col_letter = _col_to_letter(col_idx + 1)
            updates.append({"range": f"{col_letter}{target_row}",     "values": [[yhat]]})
            updates.append({"range": f"{col_letter}{target_row + 1}", "values": [[yhat_lower]]})
            updates.append({"range": f"{col_letter}{target_row + 2}", "values": [[yhat_upper]]})

    if updates:
        worksheet.batch_update(updates, value_input_option="RAW")

    written_cells = len(updates) // 3
    return {
        "written": written_cells,
        "matched": written_cells,
        "date_sample": date_sample,
        "total_cols": len([v for v in date_row_values[start_col:] if v.strip()]),
    }


def load_from_csv(filepath: str, date_col: str = "date", sales_col: str = "sales") -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df.rename(columns={date_col: "ds", sales_col: "y"})
    df["ds"] = pd.to_datetime(df["ds"], infer_datetime_format=True)
    df["y"] = pd.to_numeric(df["y"].astype(str).str.replace(",", "").str.replace("¥", ""), errors="coerce")
    df = df.dropna(subset=["ds", "y"])
    df = df[["ds", "y"]].sort_values("ds").reset_index(drop=True)
    return df


def generate_sample_data(years: int = 3) -> pd.DataFrame:
    """動作確認用サンプルデータを生成する"""
    import numpy as np

    np.random.seed(42)
    end_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    start_date = end_date - pd.Timedelta(days=365 * years)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    base = 150_000
    trend = np.linspace(0, 30_000, len(dates))
    weekly = 30_000 * np.sin(2 * np.pi * dates.dayofweek / 7 - np.pi / 2)
    # 週末は高め
    weekend_boost = np.where(dates.dayofweek >= 5, 40_000, 0)
    # 年間季節性（夏・年末年始が高い）
    yearly = 20_000 * np.sin(2 * np.pi * (dates.dayofyear - 90) / 365)
    noise = np.random.normal(0, 15_000, len(dates))

    sales = base + trend + weekly + weekend_boost + yearly + noise
    sales = np.maximum(sales, 5_000)

    df = pd.DataFrame({"ds": dates, "y": sales.round()})
    return df
