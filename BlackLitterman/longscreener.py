# long_screener.py

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf
from scipy.stats import zscore

# -------------------------------
# 1. Universe (Wikipedia scrape)
# -------------------------------
def get_wiki_table(url, ticker_col="Symbol"):
    try:
        tables = pd.read_html(url)
        for t in tables:
            if ticker_col in t.columns:
                return (
                    t[ticker_col]
                    .astype(str)
                    .str.replace(".", "-", regex=False)
                    .tolist()
                )
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
    return []

def get_universe():
    sp500 = get_wiki_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol")
    nasdaq100 = get_wiki_table("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker")
    russell2000 = get_wiki_table("https://en.wikipedia.org/wiki/List_of_Russell_2000_companies", "Ticker")
    return sorted(set(sp500 + nasdaq100 + russell2000))

universe = get_universe()
print(f"Universe size: {len(universe)}")
if not universe:
    raise ValueError("No tickers found, universe fetch failed.")

# -------------------------------
# 2. Get Fundamentals
# -------------------------------
def get_fundamentals(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "Ticker": ticker,
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Growth": info.get("earningsQuarterlyGrowth"),
            "ROE": info.get("returnOnEquity"),
            "DebtEq": info.get("debtToEquity"),
            "Beta": info.get("beta"),
            "Volatility": info.get("52WeekChange"),  # proxy for vol
        }
    except Exception:
        return None

fundamentals = []
for t in universe:
    f = get_fundamentals(t)
    if f:
        fundamentals.append(f)

df = pd.DataFrame(fundamentals)
if df.empty:
    raise ValueError("No fundamentals pulled. Yahoo Finance query failed.")

# -------------------------------
# 3. Clean + Z-Score
# -------------------------------
for col in ["Growth","ROE","DebtEq","Beta","Volatility"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df["zGrowth"] = zscore(df["Growth"].replace([float("inf"), -float("inf")], 0))
df["zROE"] = zscore(df["ROE"].replace([float("inf"), -float("inf")], 0))
df["zDebtEq"] = zscore(df["DebtEq"].replace([float("inf"), -float("inf")], 0))
df["zBeta"] = zscore(df["Beta"].replace([float("inf"), -float("inf")], 0))
df["zVol"] = zscore(df["Volatility"].replace([float("inf"), -float("inf")], 0))

# -------------------------------
# 4. LongScore
# -------------------------------
df["LongScore"] = (
    df["zGrowth"] + df["zROE"]
    - df["zDebtEq"] - df["zBeta"] - df["zVol"]
)

# -------------------------------
# 5. Rank & Output
# -------------------------------
longs = df.sort_values("LongScore", ascending=False).reset_index(drop=True)

print("\n===== Top 50 Long Candidates =====")
print(longs[["Ticker","Sector","Industry","LongScore"]].head(50))

# Save full ranking
longs.to_csv("long_candidates_full.csv", index=False)
print("\nSaved full long ranking to long_candidates_full.csv")
