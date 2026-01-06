# /Users/dave/PythonProject1/PythonProject1/PythonProject/BlackLitterman/book/analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

# === Paths ===
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
BOOK_DIR = BASE_DIR / "book"
OUT_DIR = DATA_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def safe_read_csv(path, **kwargs):
    if path.exists():
        return pd.read_csv(path, **kwargs)
    else:
        print(f"⚠️ Missing file: {path.name}")
        return pd.DataFrame()

# === Load Data ===
orth = safe_read_csv(DATA_DIR / "rolling_orthogonality.csv", index_col=0, parse_dates=True)
var_contrib = safe_read_csv(DATA_DIR / "variance_contrib.csv", index_col=0)
if not var_contrib.empty and var_contrib.shape[1] == 1:
    var_contrib = var_contrib.iloc[:, 0]  # squeeze
factor_risk = safe_read_csv(DATA_DIR / "factor_variance_contrib.csv")
port_rets = safe_read_csv(DATA_DIR / "portfolio_returns.csv", index_col=0, parse_dates=True).squeeze()
trades = safe_read_csv(BOOK_DIR / "book26082025.csv")

print("✅ Data loaded")

# === Latest orthogonality ===
last_orth = orth.iloc[-1, 0] if not orth.empty else None

# === Top variance contributors ===
top_var = var_contrib.sort_values(ascending=False).head(3) if not var_contrib.empty else None

# === Factor risk contributions ===
if not factor_risk.empty:
    row = factor_risk.iloc[0]
    top_factor = row.sort_values(ascending=False).head(3)
else:
    top_factor, row = None, None

# === VaR & ES ===
def hist_var_es(returns, alpha=0.95):
    if len(returns) == 0:
        return np.nan, np.nan
    q = np.quantile(returns, 1 - alpha)
    es = returns[returns <= q].mean()
    return q, es

var95, es95 = hist_var_es(port_rets, 0.95)
var99, es99 = hist_var_es(port_rets, 0.99)

# === Expiry ladder ===
expiry_counts = trades["expiry"].value_counts().sort_index() if not trades.empty else pd.Series()

# === Risk limit checks ===
risk_limits = {
    "portfolio_vol_cap": 0.30,   # 30% annualised
    "max_factor_share": 0.50,    # max 50% from one factor
    "min_orthogonality": 0.2     # must be >0.2
}
breaches = []
ann_vol = port_rets.std() * np.sqrt(252) if len(port_rets) > 0 else np.nan
if not np.isnan(ann_vol) and ann_vol > risk_limits["portfolio_vol_cap"]:
    breaches.append(f"Portfolio vol {ann_vol:.2%} > cap {risk_limits['portfolio_vol_cap']:.2%}")
if top_factor is not None and (top_factor.iloc[0] / row.sum()) > risk_limits["max_factor_share"]:
    breaches.append(f"Factor {top_factor.index[0]} dominates risk at {top_factor.iloc[0]/row.sum():.1%}")
if last_orth is not None and last_orth < risk_limits["min_orthogonality"]:
    breaches.append(f"Orthogonality {last_orth:.2f} < min {risk_limits['min_orthogonality']}")

# === Build Text Report ===
report = []
report.append("Portfolio Diagnostics Report")
report.append("="*28)
if last_orth is not None:
    report.append(f"Latest orthogonality score: {last_orth:.3f}")
if top_var is not None:
    report.append("\nTop variance contributors:")
    report.append(str(top_var))
if top_factor is not None:
    report.append("\nTop factor risk contributors:")
    report.append(str(top_factor))
else:
    report.append("\nTop factor risk contributors: (no factor data available)")
report.append("\nVaR/ES (Portfolio Returns):")
report.append(f"95% VaR={var95:.2%}, ES={es95:.2%} | 99% VaR={var99:.2%}, ES={es99:.2%}")
if not expiry_counts.empty:
    report.append("\nExpiry Ladder (counts per date):")
    report.append(str(expiry_counts.head(5)))
if breaches:
    report.append("\nRISK LIMIT BREACHES:")
    report.extend([f"- {b}" for b in breaches])
else:
    report.append("\nAll risk limits respected")

# Save diagnostics text
with open(OUT_DIR / "diagnostics.txt", "w") as f:
    f.write("\n".join(report))

# === Charts ===
# Factor risk bar
if top_factor is not None:
    plt.figure(figsize=(8,5))
    row.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Portfolio Risk Attribution by Factor")
    plt.ylabel("Variance Contribution")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "factor_risk_bar.png")
    plt.close()

# Rolling orthogonality
if not orth.empty:
    plt.figure(figsize=(8,5))
    orth.iloc[:,0].plot(color="darkred")
    plt.title("Rolling Orthogonality Score")
    plt.axhline(0, color="black", lw=0.8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rolling_orthogonality.png")
    plt.close()

# Traffic light dashboard
def traffic_light(value, good, warn, reverse=False):
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "grey"
    if np.isnan(val):
        return "grey"
    if reverse:
        if val > warn: return "green"
        elif val > good: return "yellow"
        else: return "red"
    else:
        if val > warn: return "green"
        elif val > good: return "yellow"
        else: return "red"

fig, ax = plt.subplots(figsize=(6,3))
ax.axis("off")
metrics = [
    ("Orthogonality", last_orth, 0.3, 0.2, False),
    ("Vol (ann.)", ann_vol, 0.25, 0.30, True),
    ("Max Factor Share", (top_factor.iloc[0]/row.sum()) if top_factor is not None else None, 0.4, 0.5, True),
]
for i, (label, val, good, warn, reverse) in enumerate(metrics):
    color = traffic_light(val, good, warn, reverse)
    ax.add_patch(plt.Rectangle((0, i*0.3), 0.2, 0.2, color=color))
    try:
        if val is not None and not np.isnan(float(val)):
            ax.text(0.25, i*0.3+0.1, f"{label}: {float(val):.2f}", va="center", fontsize=10)
    except Exception:
        pass
plt.title("Portfolio Risk Traffic Lights", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "traffic_lights.png")
plt.close()

# === PDF Pack ===
pdf_path = OUT_DIR / "portfolio_diagnostics.pdf"
with PdfPages(pdf_path) as pdf:
    # Page 1: text
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    clean_text = "\n".join([line.encode("ascii", "ignore").decode() for line in report])
    ax.text(0, 1, clean_text, ha="left", va="top", fontsize=10, family="monospace")
    pdf.savefig(fig); plt.close()
    # Add charts if available
    for chart in ["factor_risk_bar.png", "rolling_orthogonality.png", "traffic_lights.png"]:
        chart_path = OUT_DIR / chart
        if chart_path.exists():
            img = plt.imread(chart_path)
            plt.figure(figsize=(8.5, 6))
            plt.imshow(img); plt.axis("off")
            pdf.savefig(); plt.close()

print(f"\n💾 Saved all analysis + charts + PDF pack in {OUT_DIR}")
