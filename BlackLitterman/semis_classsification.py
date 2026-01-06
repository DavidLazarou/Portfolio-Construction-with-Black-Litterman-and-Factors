import pandas as pd
from pathlib import Path

# --- Step 1: Define SOXX constituents with classifications ---
data = [
    ("Advanced Micro Devices", "AMD US", "Fabless", "Historical IDM (spun off GF)"),
    ("NVIDIA", "NVDA US", "Fabless", "Pure Play"),
    ("Broadcom", "AVGO US", "Fabless", "Infrastructure/Networking mix"),
    ("Texas Instruments", "TXN US", "IDM", "Pure Play"),
    ("Qualcomm", "QCOM US", "Fabless", "Pure Play"),
    ("Monolithic Power Systems", "MPWR US", "Fabless", "Pure Play"),
    ("Micron Technology", "MU US", "IDM", "Pure Play"),
    ("Lam Research", "LRCX US", "Equipment", "Pure Play"),
    ("Intel", "INTC US", "IDM", "Foundry ambitions (Intel Foundry Services)"),
    ("Marvell Technology", "MRVL US", "Fabless", "Pure Play"),
    ("NXP Semiconductors", "NXPI US", "IDM", "Automotive/Industrial overlap"),
    ("Analog Devices", "ADI US", "IDM", "Pure Play"),
    ("KLA Corp", "KLAC US", "Equipment", "Pure Play"),
    ("Microchip Technology", "MCHP US", "IDM", "Pure Play"),
    ("Applied Materials", "AMAT US", "Equipment", "Pure Play"),
    ("Taiwan Semiconductor (TSMC)", "TSM US", "Foundry", "Advanced packaging overlap (OSAT)"),
    ("ASML Holding", "ASML US", "Equipment", "Pure Play"),
    ("ON Semiconductor", "ON US", "IDM", "Pure Play"),
    ("Teradyne", "TER US", "Equipment", "Pure Play"),
    ("Entegris", "ENTG US", "Materials", "Pure Play"),
    ("Skyworks Solutions", "SWKS US", "Fabless", "Pure Play"),
    ("Lattice Semiconductor", "LSCC US", "Fabless", "Pure Play"),
    ("Qorvo", "QRVO US", "Fabless", "Pure Play"),
    ("MKS Instruments", "MKSI US", "Equipment", "Process tools, materials overlap"),
    ("Universal Display", "OLED US", "Materials", "OLED/IP licensing"),
    ("STMicroelectronics", "STM US", "IDM", "Pure Play"),
    ("ASE Technology", "ASX US", "OSAT", "Pure Play"),
    ("Onto Innovation", "ONTO US", "Equipment", "Pure Play"),
    ("ARM Holdings", "ARM US", "EDA/IP", "Pure Play"),
    ("United Microelectronics", "UMC US", "Foundry", "Mature-node focus"),
]

# --- Step 2: Create DataFrame ---
df = pd.DataFrame(data, columns=["Company", "Ticker", "Primary Classification", "Pure Play / Crossover"])

# --- Step 3: Acronym glossary ---
acronyms = {
    "IDM": "Integrated Device Manufacturer – companies that both design and manufacture chips in-house.",
    "Fabless": "Companies that design chips but outsource manufacturing to foundries.",
    "Foundry": "Specialized manufacturers that produce chips for fabless firms.",
    "EDA": "Electronic Design Automation – providers of software tools for chip design.",
    "IP": "Intellectual Property – reusable design blocks (e.g., ARM cores).",
    "Equipment": "Makers of semiconductor manufacturing tools (e.g., lithography, etch, deposition).",
    "Materials": "Suppliers of silicon wafers, photoresists, and specialty chemicals.",
    "OSAT": "Outsourced Semiconductor Assembly and Test – packaging and testing specialists."
}

# --- Step 4: Generate LaTeX table ---
latex_table = df.to_latex(
    index=False,
    escape=False,
    caption="SOXX Constituents Classified by Semiconductor Ecosystem Role",
    label="tab:soxx_classification",
    longtable=True
)

# --- Step 5: Wrap into standalone LaTeX document ---
latex_doc = r"""\documentclass[12pt]{article}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage[margin=1in]{geometry}

\begin{document}

""" + latex_table + r"""

\section*{Glossary of Acronyms}
\begin{itemize}
""" + "\n".join([f"  \item \\textbf{{{k}}}: {v}" for k, v in acronyms.items()]) + r"""
\end{itemize}

\end{document}
"""

# --- Step 6: Save into 'tables/' folder ---
outdir = Path("tables")
outdir.mkdir(parents=True, exist_ok=True)
outfile = outdir / "soxx_classification.tex"

with open(outfile, "w") as f:
    f.write(latex_doc)

print(f"LaTeX document successfully written to {outfile}")
