# /Users/dave/PythonProject1/PythonProject1/PythonProject/BlackLitterman/book/factor_model.py

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import matplotlib.pyplot as plt
import requests, io, zipfile

# === Paths ===
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

ff_path = RAW_DIR / "ff_factors.csv"

# === Helper to clean Ken French CSV ===
def clean_ff_file(df, start_col="Unnamed: 0"):
    df = df.rename(columns=lambda x: x.strip())
    if start_col in df.columns:
        df = df.rename(columns={start_col: "Date"})
    df = df[df["Date"].astype(str).str.match(r"^\d+$")]  # keep only YYYYMMDD rows
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.set_index("Date")
    return df

# === Download Fama-French if needed ===
if not ff_path.exists():
    print("⚠️ ff_factors.csv not found, downloading from Ken French Library...")
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_Daily_CSV.zip"
    mom_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

    # FF5
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    ff5_csv = [name for name in z.namelist() if name.endswith(".csv")][0]
    ff5 = pd.read_csv(z.open(ff5_csv), skiprows=3)
    ff5 = clean_ff_file(ff5)

    # Momentum
    r2 = requests.get(mom_url)
    z2 = zipfile.ZipFile(io.BytesIO(r2.content))
    mom_csv = [name for name in z2.namelist() if name.endswith(".csv")][0]
    mom = pd.read_csv(z2.open(mom_csv), skiprows=13)
    mom = clean_ff_file(mom)
    mom = mom.rename(columns={"Mom   ": "Mom", "Mom": "Mom"})

    factors = ff5.join(mom, how="inner")
    factors.to_csv(ff_path)
    print(f"✅ Downloaded and saved Fama-French + Momentum to {ff_path}")
else:
    factors = pd.read_csv(ff_path, index_col=0, parse_dates=True)

# Convert % to decimals
factors = factors.apply(pd.to_numeric, errors="coerce") / 100.0
factors = factors.dropna()

# === Load portfolio returns ===
port_rets = pd.read_csv(DATA_DIR / "portfolio_returns.csv", index_col=0, parse_dates=True).squeeze()

# === Load daily returns for stocks ===
daily_rets = pd.read_csv(DATA_DIR / "daily_returns.csv", index_col=0, parse_dates=True)

# === Align ===
aligned = pd.concat([port_rets, factors], axis=1, join="inner").dropna()
aligned.columns = ["Portfolio"] + list(factors.columns)
aligned["Excess"] = aligned["Portfolio"] - aligned["RF"]

# === Portfolio Regression ===
X = aligned[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]]
X = sm.add_constant(X)
y = aligned["Excess"]
model = sm.OLS(y, X).fit()
print("✅ Factor regression results (Portfolio):")
print(model.summary())

# Save portfolio loadings
port_loadings = pd.DataFrame({
    "Factor": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
    "Beta": model.params[1:],
    "t-stat": model.tvalues[1:]
})
port_loadings.to_csv(DATA_DIR / "factor_loadings_portfolio.csv", index=False)

# === Per-stock regressions ===
stock_results = []
for ticker in daily_rets.columns:
    tmp = pd.concat([daily_rets[ticker], factors], axis=1, join="inner").dropna()
    tmp["Excess"] = tmp[ticker] - tmp["RF"]
    Xs = sm.add_constant(tmp[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]])
    ys = tmp["Excess"]
    res = sm.OLS(ys, Xs).fit()
    for fac in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]:
        stock_results.append({
            "Ticker": ticker,
            "Factor": fac,
            "Beta": res.params[fac],
            "t-stat": res.tvalues[fac]
        })

stock_df = pd.DataFrame(stock_results)
stock_df.to_csv(DATA_DIR / "factor_loadings_stocks.csv", index=False)
print(f"✅ Saved per-stock factor loadings to {DATA_DIR / 'factor_loadings_stocks.csv'}")

# === Factor Contribution to Variance (portfolio) ===
factor_cov = factors[["Mkt-RF","SMB","HML","RMW","CMA","Mom"]].cov().values
beta_vec = model.params[1:].values.reshape(-1,1)

var_factor = float(beta_vec.T @ factor_cov @ beta_vec)
var_total = float(np.var(y))
var_idio = var_total - var_factor

# Per-factor contribution (vectorized)
factor_contrib = (beta_vec.flatten() * (factor_cov @ beta_vec).flatten())

risk_contrib = {
    "Mkt-RF": factor_contrib[0],
    "SMB": factor_contrib[1],
    "HML": factor_contrib[2],
    "RMW": factor_contrib[3],
    "CMA": factor_contrib[4],
    "Mom": factor_contrib[5],
    "Idio": var_idio
}

risk_df = pd.DataFrame([risk_contrib])
risk_df.to_csv(DATA_DIR / "factor_variance_contrib.csv", index=False)
print(f"✅ Saved variance attribution to {DATA_DIR / 'factor_variance_contrib.csv'}")

# Pie chart of risk attribution
plt.figure(figsize=(6,6))
plt.pie(risk_df.iloc[0], labels=risk_df.columns, autopct="%.1f%%")
plt.title("Portfolio Risk Attribution (Factor vs Idio)")
plt.savefig(DATA_DIR / "factor_risk_attribution.png")
plt.close()

# === Rolling Portfolio Betas ===
window = 120  # 6 months
roll_betas = []
dates = []

for i in range(window, len(aligned)):
    sub = aligned.iloc[i-window:i]
    Xr = sm.add_constant(sub[["Mkt-RF","SMB","HML","RMW","CMA","Mom"]])
    yr = sub["Excess"]
    res = sm.OLS(yr, Xr).fit()
    roll_betas.append(res.params[1:].values)
    dates.append(aligned.index[i])

roll_df = pd.DataFrame(roll_betas, index=dates, columns=["Mkt-RF","SMB","HML","RMW","CMA","Mom"])
roll_df.to_csv(DATA_DIR / "rolling_betas_portfolio.csv")

plt.figure(figsize=(10,6))
for col in roll_df.columns:
    plt.plot(roll_df.index, roll_df[col], label=col)
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Rolling Portfolio Betas (120d)")
plt.legend()
plt.tight_layout()
plt.savefig(DATA_DIR / "rolling_betas_portfolio.png")
plt.close()

print(f"💾 Saved all outputs in {DATA_DIR}")
