# book/orthogonal.py

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# === Directories ===
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# === Load returns ===
returns = pd.read_csv(PROC_DIR / "daily_returns.csv", index_col=0, parse_dates=True)

# === Load factors from RAW (not processed) ===
ff_path = RAW_DIR / "ff_factors.csv"
if not ff_path.exists():
    raise FileNotFoundError(f"❌ Missing {ff_path}. Run factor_model.py first.")

factors = pd.read_csv(ff_path, index_col=0, parse_dates=True) / 100
factors["ExcessMkt"] = factors["Mkt-RF"]

# === Settings ===
window = 126  # 6 months rolling
tickers = returns.columns

rolling_betas = {}
rolling_var_attrib = []
rolling_orthog = []
rolling_vol = []

# === Rolling regressions per stock ===
for ticker in tickers:
    df = pd.concat([returns[ticker], factors], axis=1).dropna()
    df.rename(columns={ticker: "ret"}, inplace=True)

    betas = []
    for i in range(window, len(df)):
        y = df["ret"].iloc[i - window:i].values - factors["RF"].iloc[i - window:i].values
        X = df[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]].iloc[i - window:i].values
        X = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        betas.append(beta)

    betas = pd.DataFrame(
        betas,
        index=df.index[window:],
        columns=["alpha", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
    )
    rolling_betas[ticker] = betas

    for date in betas.index:
        b = betas.loc[date].values[1:]
        fcov = factors.loc[:date, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]].cov()
        var_factor = float(b.T @ fcov.values @ b)
        var_total = np.var(df["ret"].loc[:date].tail(window))
        var_idio = max(var_total - var_factor, 0)
        rolling_var_attrib.append(
            {
                "Date": date,
                "Ticker": ticker,
                "Mkt-RF": b[0] ** 2 * fcov.values[0, 0],
                "SMB": b[1] ** 2 * fcov.values[1, 1],
                "HML": b[2] ** 2 * fcov.values[2, 2],
                "RMW": b[3] ** 2 * fcov.values[3, 3],
                "CMA": b[4] ** 2 * fcov.values[4, 4],
                "Mom": b[5] ** 2 * fcov.values[5, 5],
                "Idio": var_idio,
            }
        )

# === Portfolio-level ===
port_rets = returns.mean(axis=1)
dfp = pd.concat([port_rets, factors], axis=1).dropna()
dfp.rename(columns={0: "ret"}, inplace=True)

for i in range(window, len(dfp)):
    y = dfp["ret"].iloc[i - window:i].values - factors["RF"].iloc[i - window:i].values
    X = dfp[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]].iloc[i - window:i].values
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X, y, rcond=None)[0][1:]

    fcov = factors.loc[: dfp.index[i], ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]].cov()
    var_factor = float(beta.T @ fcov.values @ beta)
    var_total = np.var(dfp["ret"].iloc[i - window:i])
    var_idio = max(var_total - var_factor, 0)

    rolling_orthog.append({"Date": dfp.index[i], "Orthogonality": var_idio / var_total})
    rolling_vol.append({"Date": dfp.index[i], "Vol": np.sqrt(var_total * 252)})

# === DataFrames ===
var_attrib_df = pd.DataFrame(rolling_var_attrib).set_index("Date")
orthog_df = pd.DataFrame(rolling_orthog).set_index("Date")
vol_df = pd.DataFrame(rolling_vol).set_index("Date")

# === Save ===
var_attrib_df.to_csv(PROC_DIR / "factor_variance_attrib.csv")
orthog_df.to_csv(PROC_DIR / "orthogonality_series.csv")
vol_df.to_csv(PROC_DIR / "portfolio_vol_series.csv")

# === Visuals ===
plt.figure(figsize=(12, 6))
var_attrib_df.groupby("Date")[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "Idio"]].mean().plot.area(
    stacked=True, alpha=0.8, ax=plt.gca()
)
plt.title("Rolling Variance Attribution (Portfolio avg)")
plt.ylabel("Variance share")
plt.tight_layout()
plt.savefig(PROC_DIR / "variance_attribution_stacked.png")
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(orthog_df.index, orthog_df["Orthogonality"], label="Orthogonality (Idio %)")
plt.plot(vol_df.index, vol_df["Vol"], label="Portfolio Vol (ann.)", alpha=0.7)
plt.legend()
plt.title("Orthogonality vs Portfolio Volatility")
plt.tight_layout()
plt.savefig(PROC_DIR / "orthogonality_vs_vol.png")
plt.close()

print("✅ Rolling betas computed and saved per stock")
print(f"✅ Rolling orthogonality score saved (last={orthog_df['Orthogonality'].iloc[-1]:.3f})")
print(f"💾 Saved all outputs in {PROC_DIR}")
