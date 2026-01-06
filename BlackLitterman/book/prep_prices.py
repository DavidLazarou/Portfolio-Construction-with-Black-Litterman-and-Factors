# /Users/dave/PythonProject1/PythonProject1/PythonProject/BlackLitterman/book/prep_prices.py

import pandas as pd
import yfinance as yf
from pathlib import Path

# === Input: your trade blotter ===
book_file = Path(__file__).parent / "book26082025.csv"
book = pd.read_csv(book_file)

# Extract unique tickers
tickers = book["ticker"].unique().tolist()

# === Output path ===
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)
output_file = DATA_DIR / "equity_prices.csv"

# === Download daily prices ===
print(f"📥 Downloading {len(tickers)} tickers: {tickers}")

# Explicitly set auto_adjust=False so Adj Close is returned
prices = yf.download(tickers, start="2020-01-01", auto_adjust=False)

# Handle column structure (multi vs single index)
if isinstance(prices.columns, pd.MultiIndex):
    if "Adj Close" in prices.columns.levels[0]:
        prices = prices["Adj Close"]
    elif "Close" in prices.columns.levels[0]:
        prices = prices["Close"]
else:
    if "Adj Close" in prices.columns:
        prices = prices[["Adj Close"]]
    elif "Close" in prices.columns:
        prices = prices[["Close"]]

# Save
prices.to_csv(output_file)
print(f"✅ Saved equity prices to {output_file}")
