# /Users/dave/PythonProject1/PythonProject1/PythonProject/BlackLitterman/book/stress_test.py

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

try:
    import QuantLib as ql
    HAS_QL = True
except ImportError:
    HAS_QL = False

# === Paths ===
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
OUT_DIR = DATA_DIR / "stress_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === Inputs ===
trades = pd.read_csv(BASE_DIR / "book" / "book26082025.csv")
try:
    prices = pd.read_csv(DATA_DIR / "prices.csv", index_col=0, parse_dates=True)
except FileNotFoundError:
    print("⚠️ prices.csv not found, falling back to daily_returns.csv approximation...")
    prices = pd.read_csv(DATA_DIR / "daily_returns.csv", index_col=0, parse_dates=True).cumsum()

spot_dict = prices.iloc[-1].to_dict()
r = 0.0428  # fallback T-bill yield
div_yield = 0.0  # assume no dividends for now
vol_guess = 0.5  # fallback vol (50%)

# === Stress scenarios ===
scenarios = {
    "Spot -20%": {"spot": 0.8, "vol": 1.0, "days": 0},
    "Spot +20%": {"spot": 1.2, "vol": 1.0, "days": 0},
    "Vol -20%": {"spot": 1.0, "vol": 0.8, "days": 0},
    "Vol +20%": {"spot": 1.0, "vol": 1.2, "days": 0},
    "Time -10d": {"spot": 1.0, "vol": 1.0, "days": -10},
}

# === Option pricer ===
def price_option_qllib(row, spot, vol, days_shift):
    try:
        expiry = pd.to_datetime(row["expiry"], dayfirst=True)
    except Exception:
        expiry = pd.to_datetime(row["expiry"])
    today = pd.Timestamp.today()
    maturity = (expiry - today).days / 365.0 + days_shift/365.0
    if maturity <= 0:
        return {"price": 0, "delta": 0, "gamma": 0, "vega": 0, "theta": 0}

    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if row["callput"].lower()=="call" else ql.Option.Put,
        row["strike"]
    )
    exercise = ql.EuropeanExercise(ql.Date(expiry.day, expiry.month, expiry.year))
    opt = ql.VanillaOption(payoff, exercise)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), r, ql.Actual365Fixed()))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), div_yield, ql.Actual365Fixed()))
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, ql.NullCalendar(), vol, ql.Actual365Fixed()))

    bsm = ql.BlackScholesMertonProcess(spot_handle, div_ts, flat_ts, vol_ts)
    engine = ql.AnalyticEuropeanEngine(bsm)
    opt.setPricingEngine(engine)

    return {
        "price": opt.NPV(),
        "delta": opt.delta(),
        "gamma": opt.gamma(),
        "vega": opt.vega(),
        "theta": opt.theta()
    }

# === Stress loop ===
results = []
for i, row in trades.iterrows():
    ticker = row["ticker"]
    base_spot = spot_dict.get(ticker, np.nan)
    if pd.isna(base_spot):
        continue

    for scen, shocks in scenarios.items():
        spot = base_spot * shocks["spot"]
        vol = vol_guess * shocks["vol"]
        days = shocks["days"]

        if HAS_QL:
            res = price_option_qllib(row, spot, vol, days)
        else:
            # fallback to zero greeks if no QuantLib
            res = {"price": 0, "delta": 0, "gamma": 0, "vega": 0, "theta": 0}

        pnl = (res["price"] - row["mkt"]) * row["contracts"]
        results.append({
            "Ticker": ticker,
            "Scenario": scen,
            "PnL": pnl,
            "Delta": res["delta"] * row["contracts"],
            "Gamma": res["gamma"] * row["contracts"],
            "Vega": res["vega"] * row["contracts"],
            "Theta": res["theta"] * row["contracts"]
        })

df = pd.DataFrame(results)
df.to_csv(OUT_DIR / "stress_scenarios.csv", index=False)

# === Portfolio-level summary ===
summary = df.groupby("Scenario")[["PnL","Delta","Gamma","Vega","Theta"]].sum()
summary.to_csv(OUT_DIR / "stress_summary.csv")

# === Visuals ===
plt.figure(figsize=(10,6))
summary["PnL"].plot(kind="bar", color="orange")
plt.axhline(0, color="black")
plt.title("Portfolio Stressed P&L")
plt.ylabel("PnL")
plt.tight_layout()
plt.savefig(OUT_DIR / "stress_pnl.png")
plt.close()

plt.figure(figsize=(10,6))
summary[["Delta","Gamma","Vega","Theta"]].plot(kind="bar")
plt.title("Portfolio Greeks under Stress Scenarios")
plt.tight_layout()
plt.savefig(OUT_DIR / "stress_greeks.png")
plt.close()

print(f"✅ Stress tests complete. Saved results in {OUT_DIR}")
