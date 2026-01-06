# /Users/dave/PythonProject1/PythonProject1/PythonProject/BlackLitterman/book/create_equity_prices.py

import pandas as pd
import yfinance as yf
from pathlib import Path

# === Input book file ===
book_file = Path(__file__).parent / "book26082025.csv"
tickers = pd.read_csv(book_file)["Ticker"].dropna().unique().tolist()

# === Output directory ===
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

output_file = DATA_DIR / "equity_prices.csv"

print(f"📥 Downloading prices for {len(tickers)} tickers...")
prices = yf.download(tickers, start="2018-01-01")["Adj Close"]
prices.to_csv(output_file)

print(f"✅ Saved equity prices to {output_file}")
