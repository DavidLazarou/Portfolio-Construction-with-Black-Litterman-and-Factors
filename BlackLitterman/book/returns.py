# /Users/dave/PythonProject1/PythonProject1/PythonProject/BlackLitterman/book/returns.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA

# === Paths ===
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# === Load Prices ===
prices_path = DATA_DIR / "equity_prices.csv"
prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)

# === Compute Returns ===
returns = prices.pct_change().dropna()
print("✅ Daily returns calculated")

# === Portfolio Volatility ===
cov_matrix = returns.cov()
weights = np.repeat(1 / returns.shape[1], returns.shape[1])  # equal weights
port_var = np.dot(weights.T, np.dot(cov_matrix.values, weights))
port_vol = np.sqrt(port_var) * np.sqrt(252)
print(f"✅ Portfolio volatility (ann.): {port_vol:.2%}")

# === Correlations ===
corr_matrix = returns.corr()
mean_offdiag_corr = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool)).mean().mean()
print(f"✅ Mean off-diagonal correlation: {mean_offdiag_corr:.2%}")

# === Variance Contributions ===
marginal_contrib = np.dot(cov_matrix.values, weights)
risk_contrib = weights * marginal_contrib
var_contrib = risk_contrib / port_var
variance_contrib = pd.Series(var_contrib, index=returns.columns)
print("\n✅ Variance contribution by ticker:")
print(variance_contrib)

# === PCA ===
pca = PCA()
pca.fit(returns)

explained_var = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
    "ExplainedVariance": pca.explained_variance_ratio_,
    "Cumulative": np.cumsum(pca.explained_variance_ratio_)
})
print("\n✅ PCA Explained Variance:")
print(explained_var.head())

loadings = pd.DataFrame(
    pca.components_.T,
    index=returns.columns,
    columns=[f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))]
)
print("\n✅ PCA Loadings (first 5 PCs):")
print(loadings.iloc[:, :5])

# === Save Outputs ===
returns.to_csv(DATA_DIR / "daily_returns.csv")
returns.mean(axis=1).to_csv(DATA_DIR / "portfolio_returns.csv")
corr_matrix.to_csv(DATA_DIR / "correlation_matrix.csv")
variance_contrib.to_csv(DATA_DIR / "variance_contrib.csv")
explained_var.to_csv(DATA_DIR / "pca_explained_variance.csv", index=False)
loadings.to_csv(DATA_DIR / "pca_loadings.csv")

# === Visualisations ===
plt.style.use("seaborn-v0_8")

# 1. Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(DATA_DIR / "corr_heatmap.png")
plt.close()

# 2. Variance contribution bar chart
variance_contrib.sort_values(ascending=False).plot(
    kind="bar", figsize=(8, 4), title="Variance Contribution by Ticker"
)
plt.ylabel("Proportion of Total Variance")
plt.tight_layout()
plt.savefig(DATA_DIR / "variance_contrib.png")
plt.close()

# 3. PCA scree plot (cumulative explained variance)
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         explained_var["Cumulative"], marker="o")
plt.xlabel("Principal Component")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Scree Plot")
plt.grid(True)
plt.tight_layout()
plt.savefig(DATA_DIR / "pca_scree.png")
plt.close()

# 4. PCA loadings heatmap (first 5 PCs)
plt.figure(figsize=(10, 6))
sns.heatmap(loadings.iloc[:, :5], annot=True, cmap="coolwarm", center=0)
plt.title("PCA Loadings (First 5 PCs)")
plt.tight_layout()
plt.savefig(DATA_DIR / "pca_loadings.png")
plt.close()

# 5. Cumulative portfolio return
cum_portfolio = (returns.mean(axis=1) + 1).cumprod()
plt.figure(figsize=(10, 4))
cum_portfolio.plot(title="Cumulative Portfolio Return")
plt.ylabel("Growth of $1 (Equal Weighted)")
plt.tight_layout()
plt.savefig(DATA_DIR / "cum_portfolio_return.png")
plt.close()

# 6. Rolling 60-day vol
rolling_vol = returns.mean(axis=1).rolling(60).std() * np.sqrt(252)
rolling_vol.to_csv(DATA_DIR / "rolling_vol.csv")
plt.figure(figsize=(10, 4))
rolling_vol.plot(title="Rolling 60-day Portfolio Volatility")
plt.ylabel("Annualised Volatility")
plt.tight_layout()
plt.savefig(DATA_DIR / "rolling_vol.png")
plt.close()

# 7. Rolling 60-day mean correlation
rolling_mean_corr = []
for i in range(60, len(returns)):
    window = returns.iloc[i-60:i]
    corr = window.corr().values
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
    rolling_mean_corr.append(off_diag.mean())

rolling_mean_corr = pd.Series(
    rolling_mean_corr, index=returns.index[60:], name="RollingMeanCorr"
)
rolling_mean_corr.to_csv(DATA_DIR / "rolling_mean_corr.csv")

plt.figure(figsize=(10, 4))
rolling_mean_corr.plot(title="Rolling 60-day Mean Off-diagonal Correlation")
plt.ylabel("Correlation")
plt.tight_layout()
plt.savefig(DATA_DIR / "rolling_mean_corr.png")
plt.close()

print("\n💾 Saved CSV + PNGs in:", DATA_DIR)
