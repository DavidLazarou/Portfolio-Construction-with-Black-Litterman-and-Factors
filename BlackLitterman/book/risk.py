# risk.py

import os
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
from pandas_datareader import data as pdr

# --- Config ---
FALLBACK_TBILL = 4.28  # fallback rate if FRED fails

# --- Data directories ---
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
os.makedirs(DATA_DIR, exist_ok=True)


class DataIngest:
    def __init__(self):
        pass

    # -------------------------
    # 0. Load trades book
    # -------------------------
    def load_trades(self, file_path: str):
        trades = pd.read_csv(file_path)
        trades["expiry"] = pd.to_datetime(trades["expiry"], dayfirst=True, errors="coerce")
        trades.to_csv(DATA_DIR / "trades_clean.csv", index=False)
        return trades

    # -------------------------
    # 1. Equity prices
    # -------------------------
    def get_equity_prices(self, tickers, start="2022-01-01", end=None):
        end = end or datetime.today().strftime("%Y-%m-%d")
        try:
            prices = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=True,
                threads=True
            )["Close"]
            prices.to_csv(DATA_DIR / "equity_prices.csv")
            return prices
        except Exception as e:
            print(f"⚠️ Equity price fetch failed: {e}")
            return pd.DataFrame()

    # -------------------------
    # 2. Dividends
    # -------------------------
    def get_dividends(self, ticker):
        try:
            t = yf.Ticker(ticker)
            divs = t.dividends
            divs.to_csv(DATA_DIR / f"{ticker}_dividends.csv")
            return divs
        except Exception as e:
            print(f"⚠️ Could not fetch dividends for {ticker}: {e}")
            return pd.Series(dtype=float)

    # -------------------------
    # 3. T-bill rates
    # -------------------------
    def get_tbill_rates(self, start="2022-01-01"):
        """
        Fetch US T-Bill rates using pandas_datareader only.
        If fails, return a constant fallback (4.28%).
        """
        series_codes = {"DTB3": "3M", "DTB6": "6M", "DTB12": "1Y"}
        df = pd.DataFrame()

        try:
            for code, label in series_codes.items():
                df[label] = pdr.DataReader(code, "fred", start=start)
            print("✅ T-bill rates fetched")
        except Exception as e:
            print(f"⚠️ Could not fetch T-bill rates from FRED: {e}")
            dates = pd.date_range(start=start, end=datetime.today(), freq="B")
            df = pd.DataFrame({"1Y": [FALLBACK_TBILL] * len(dates)}, index=dates)
            print(f"⚠️ Using fallback T-bill rate = {FALLBACK_TBILL}%")

        df.to_csv(DATA_DIR / "tbill_rates.csv")
        return df


if __name__ == "__main__":
    trade_file = "/data/raw/book26082025.csv"

    ingest = DataIngest()

    # Trades
    trades = ingest.load_trades(trade_file)
    print("✅ Trades loaded:")
    print(trades.head())

    tickers = trades["ticker"].unique().tolist()
    print(f"Unique tickers: {tickers}")

    # Prices
    prices = ingest.get_equity_prices(tickers, start="2023-01-01")
    if not prices.empty:
        print("✅ Equity prices fetched")

    # Dividends
    divs = ingest.get_dividends("SBUX")
    if not divs.empty:
        print("✅ Dividend series for SBUX:", divs.tail())

    # T-bills
    tbills = ingest.get_tbill_rates()
    print("T-bill rates (tail):")
    print(tbills.tail())
