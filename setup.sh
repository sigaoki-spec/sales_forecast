#!/bin/bash
set -e

echo "=== 売上予測アプリ セットアップ ==="

# Python バージョン確認
python3 --version

# 仮想環境作成
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  echo "仮想環境を作成しました"
fi

source .venv/bin/activate

# 依存パッケージインストール
pip install --upgrade pip
pip install -r requirements.txt

# .env ファイルの準備
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo ".env ファイルを作成しました。内容を編集してください。"
fi

echo ""
echo "=== セットアップ完了 ==="
echo "起動コマンド: source .venv/bin/activate && streamlit run app.py"
