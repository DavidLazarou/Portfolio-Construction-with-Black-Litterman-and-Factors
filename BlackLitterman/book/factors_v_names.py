import pandas as pd
import statsmodels.api as sm
import yfinance as yf

# --- 1. Download stock data ---
ticker = "XBI"
stock = yf.download(ticker, start="2020-01-01", end="2025-01-01")

# Handle flat vs MultiIndex columns, preferring Adj Close, else Close
if isinstance(stock.columns, pd.MultiIndex):
    if ("Adj Close", ticker) in stock.columns:
        stock = stock[("Adj Close", ticker)]
    elif ("Close", ticker) in stock.columns:
        stock = stock[("Close", ticker)]
    else:
        raise KeyError(f"No Adj Close or Close found in MultiIndex columns: {stock.columns}")
else:
    if "Adj Close" in stock.columns:
        stock = stock["Adj Close"]
    elif "Close" in stock.columns:
        stock = stock["Close"]
    else:
        raise KeyError(f"No Adj Close or Close found in flat columns: {stock.columns}")

returns = stock.pct_change().dropna()

# --- 2. Download Fama-French 5 factors ---
url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
ff = pd.read_csv(url, skiprows=3)

# --- Clean footer / non-numeric rows ---
# Keep only rows where 'Mkt-RF' is a number
ff = ff[pd.to_numeric(ff["Mkt-RF"], errors="coerce").notna()]

# Convert all to numeric
ff = ff.apply(pd.to_numeric, errors="coerce")

# Set index to datetime
ff.index = pd.to_datetime(ff["Unnamed: 0"].astype(str), format="%Y%m%d")
ff = ff.drop(columns=["Unnamed: 0"])

# --- 3. Align stock returns and factors ---
ff = ff.loc[returns.index]   # align by trading dates
y = (returns * 100 - ff["RF"])  # stock excess return (%)
X = ff.drop(columns=["RF"])
X = sm.add_constant(X)

# --- 4. Run regression ---
model = sm.OLS(y, X).fit()

# --- 5. Print results ---
print(model.summary())

# Nicely formatted table of betas + t-stats
results = pd.DataFrame({
    "coef": model.params,
    "tstat": model.tvalues,
    "pval": model.pvalues
})
print("\n=== Factor Loadings ===")
print(results.round(4))
