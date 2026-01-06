# factors.py

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt

# -------------------------------
# 1. Define long and candidate short books
# -------------------------------
longs = ["DECK", "MOH", "CFLT", "CPER", "XBI"]

# Expanded candidate short universe (can swap in your watchlist)
candidate_shorts = [
    "SBUX", "TXN", "SLB", "CCL", "AAL", "F", "UAL", "RCL", "DKNG", "CVNA"
]

raw = yf.download(longs + candidate_shorts, start="2015-01-01", end="2025-01-01",
                  interval="1mo", auto_adjust=True, progress=False)
prices = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']]

# -------------------------------
# 2. Compute returns
# -------------------------------
returns = prices.pct_change(fill_method=None).dropna()
returns['Longs'] = returns[longs].mean(axis=1)

# -------------------------------
# 3. Fama-French 5 + Momentum
# -------------------------------
ff5 = pdr.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench",
                     start=2015, end=2025)[0]
momentum = pdr.DataReader("F-F_Momentum_Factor", "famafrench",
                          start=2015, end=2025)[0]
momentum = momentum.rename(columns={momentum.columns[0]: "Mom"})

factors = ff5.join(momentum, how="inner")
factors.index = factors.index.to_timestamp()
factors = factors / 100

# -------------------------------
# 4. Macro Factors
# -------------------------------
macro_tickers = ["USO", "TLT", "GLD", "^VIX", "^TNX", "^IRX"]
macro_raw = yf.download(macro_tickers, start="2015-01-01", end="2025-01-01",
                        interval="1mo", auto_adjust=True, progress=False)['Close']
macro_returns = macro_raw.pct_change(fill_method=None)

yields = yf.download(["^TNX","^IRX"], start="2015-01-01", end="2025-01-01",
                     interval="1mo", progress=False)['Close']
yields = yields.dropna()
spreads = (yields["^TNX"] - yields["^IRX"]).rename("2s10s")

macro = pd.concat([macro_returns[['USO','TLT','GLD','^VIX']],
                   spreads.pct_change(fill_method=None)], axis=1)
macro = macro.rename(columns={"^VIX":"VIX","2s10s":"2s10s"})

# -------------------------------
# 5. Combine All Factors
# -------------------------------
returns.index = returns.index.to_period("M").to_timestamp()
all_factors = factors.join(macro, how="inner").dropna()
data = returns.join(all_factors, how="inner").dropna()

rf = data['RF']
excess_returns = data[longs + candidate_shorts + ['Longs']].sub(rf, axis=0)

# -------------------------------
# 6. Regression Helper
# -------------------------------
X_cols = ['Mkt-RF','SMB','HML','RMW','CMA','Mom','USO','TLT','GLD','VIX','2s10s']

def run_regression(y, factors):
    X = factors[X_cols].copy()
    X = sm.add_constant(X, has_constant="add")
    df = pd.concat([y, X], axis=1).dropna()
    return sm.OLS(df.iloc[:, 0], df.iloc[:, 1:]).fit()

# -------------------------------
# 7. Screen Shorts by RMW & CMA
# -------------------------------
screen_table = []

for ticker in candidate_shorts:
    model = run_regression(excess_returns[ticker], data)
    params = model.params
    screen_table.append({
        "Ticker": ticker,
        "Alpha": params.get("const", 0),
        "RMW": params.get("RMW", 0),
        "CMA": params.get("CMA", 0),
        "R2": model.rsquared
    })

screen_df = pd.DataFrame(screen_table)
screen_df = screen_df.sort_values(by=["RMW","CMA"], ascending=[True, True])  # lowest first

print("\n===== Candidate Short Ranking (Lowest Profitability & Investment Quality) =====")
print(screen_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# -------------------------------
# 8. Select Short Basket
# -------------------------------
shorts = screen_df.head(3)["Ticker"].tolist()  # top 3 weakest
print(f"\nSelected Shorts: {shorts}")

returns['Shorts'] = returns[shorts].mean(axis=1)
returns['Portfolio'] = returns['Longs'] - returns['Shorts']

# -------------------------------
# 9. Portfolio Cumulative P&L
# -------------------------------
cum_rets = (1 + returns[['Longs','Shorts','Portfolio']]).cumprod()
cum_rets.to_csv("long_short_pnls.csv")

plt.figure(figsize=(10,6))
plt.plot(cum_rets['Longs'], label="Longs", color="green")
plt.plot(cum_rets['Shorts'], label="Shorts", color="red")
plt.plot(cum_rets['Portfolio'], label="Portfolio (L-S)", color="black")
plt.title("Cumulative P&Ls with Screened Shorts")
plt.xlabel("Date"); plt.ylabel("Growth of $1")
plt.legend(); plt.grid(True); plt.show()
