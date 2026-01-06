# longshortscreener.py

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from pandas_datareader import data as pdr
import time

# -------------------------------
# 1. Get Universe from Wikipedia
# -------------------------------
def get_wiki_table(url, ticker_col="Symbol"):
    try:
        tables = pd.read_html(url)
        for t in tables:
            if ticker_col in t.columns:
                tickers = (
                    t[ticker_col]
                    .astype(str)
                    .str.replace(".", "-", regex=False)
                    .tolist()
                )
                return tickers
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
    return []

def get_universe():
    sp500 = get_wiki_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol")
    nasdaq100 = get_wiki_table("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker")
    russell2000 = get_wiki_table("https://en.wikipedia.org/wiki/List_of_Russell_2000_companies", "Ticker")
    all_tickers = sp500 + nasdaq100 + russell2000
    return sorted(set(all_tickers))

universe = get_universe()
print(f"Universe size: {len(universe)}")
if not universe:
    raise ValueError("Universe fetch failed. Wikipedia scrape issue.")

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
            "PE": info.get("forwardPE"),
            "ROE": info.get("returnOnEquity"),
            "PMargin": info.get("profitMargins"),
            "DebtEq": info.get("debtToEquity"),
            "Growth": info.get("earningsQuarterlyGrowth"),
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
# 3. Fundamental Scoring
# -------------------------------
for col in ["PE","ROE","PMargin","DebtEq","Growth"]:
    if col not in df.columns:
        df[col] = 0
df = df.fillna(0)

df["FundamentalScore"] = (
    df["ROE"].fillna(0) +
    df["PMargin"].fillna(0) +
    df["Growth"].fillna(0) -
    df["PE"].rank(pct=True) -
    df["DebtEq"].rank(pct=True)
)

# -------------------------------
# 4. Fama-French 5 + Momentum
# -------------------------------
ff5 = pdr.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench",
                     start=1995, end=2025)[0]
momentum = pdr.DataReader("F-F_Momentum_Factor", "famafrench",
                          start=1995, end=2025)[0]
momentum = momentum.rename(columns={momentum.columns[0]: "Mom"})
factors = ff5.join(momentum, how="inner")
factors.index = factors.index.to_timestamp()
factors = factors / 100
X_cols = ["Mkt-RF","SMB","HML","RMW","CMA","Mom"]

# -------------------------------
# 5. Factor Regression (batched)
# -------------------------------
def factor_regression(ticker):
    try:
        px = yf.download(ticker, start="2018-01-01", end="2025-01-01",
                         interval="1mo", auto_adjust=True, progress=False)["Close"]
        rets = px.pct_change().dropna()
        rets.index = rets.index.to_period("M").to_timestamp()
        data = pd.concat([rets, factors], axis=1).dropna()
        if data.empty:
            return None
        y = data[ticker] - data["RF"]
        X = sm.add_constant(data[X_cols])
        model = sm.OLS(y, X).fit()
        return model.params
    except Exception:
        return None

factor_results = {}
batch_size = 200
for i in range(0, len(df), batch_size):
    batch = df["Ticker"].iloc[i:i+batch_size]
    for t in batch:
        params = factor_regression(t)
        if params is not None:
            factor_results[t] = params
    print(f"Processed {i+len(batch)} / {len(df)} tickers")
    time.sleep(2)  # avoid rate limiting

# -------------------------------
# 6. Merge Scores
# -------------------------------
scores = []
for t, params in factor_results.items():
    row = df[df["Ticker"]==t].iloc[0]
    fund = row["FundamentalScore"]
    rmw = params.get("RMW",0)
    cma = params.get("CMA",0)
    factor_score = rmw - cma
    combined = fund + factor_score
    scores.append({
        "Ticker":t,
        "Sector":row["Sector"],
        "Industry":row["Industry"],
        "FundamentalScore":fund,
        "FactorScore":factor_score,
        "CombinedScore":combined
    })

scores_df = pd.DataFrame(scores).sort_values("CombinedScore", ascending=False)

# -------------------------------
# 7. Quantile Buckets (Deciles)
# -------------------------------
scores_df["Decile"] = pd.qcut(scores_df["CombinedScore"], 10, labels=False, duplicates="drop") + 1

scores_df["Bucket"] = "Neutral"
scores_df.loc[scores_df["Decile"] >= 9, "Bucket"] = "Top Longs"
scores_df.loc[scores_df["Decile"] <= 2, "Bucket"] = "Top Shorts"

# -------------------------------
# 8. Output Long/Short Lists
# -------------------------------
longs = scores_df[scores_df["Bucket"]=="Top Longs"]
shorts = scores_df[scores_df["Bucket"]=="Top Shorts"]

print(f"\n===== Top Longs (Decile 9-10) =====")
print(longs.head(30))

print(f"\n===== Top Shorts (Decile 1-2) =====")
print(shorts.head(30))

# Save all
scores_df.to_csv("long_short_full_ranking.csv", index=False)
longs.to_csv("long_candidates.csv", index=False)
shorts.to_csv("short_candidates.csv", index=False)

print("\nSaved longs, shorts, and full ranking to CSV.")
