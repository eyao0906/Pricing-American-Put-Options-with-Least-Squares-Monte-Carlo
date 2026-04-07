"""
run_asian_study.py
------------------
Main script for the American-Bermuda-Asian option comparison study.

Prices the ABA option across a range of (S0, sigma) combinations
matching the LSM paper Table 3 setup, then compares:
  1. LSM (Longstaff-Schwartz) with Laguerre basis, ITM-only regression
  2. TVR-style fitted value iteration with polynomial basis, all-path regression
  3. European Asian MC benchmark (lower bound)

Output:
  - Pricing table printed to console
  - CSV saved to data/asian_pricing_results.csv
  - Comparison figure saved to visuals/asian_model_comparison.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Make sure imports work whether run from project root or src/
sys.path.insert(0, os.path.dirname(__file__))

from asian_lsmc import price_aba_lsmc, price_european_asian_mc
from asian_tvr  import price_aba_tvr


# ---------------------------------------------------------------------------
# Experiment parameters (following LSM paper Table 3 spirit)
# ---------------------------------------------------------------------------

# Fixed parameters
r       = 0.06
T       = 2.0
lockout = 0.25
K       = 100.0
n_steps = 200      # 100 per year, matching LSM paper
n_paths = 50000    # 25,000 + 25,000 antithetic, matching LSM paper
seed    = 42

# Grid of (S0, sigma) combinations
# LSM Table 3 uses A (initial average) and S around 90, 100, 110
# and sigma = 0.20 fixed. We vary S0 and keep sigma = 0.20.
S0_values    = [90.0, 100.0, 110.0]
sigma_values = [0.20]   # LSM Table 3 fixes sigma = 0.20

# ---------------------------------------------------------------------------
# Run pricing
# ---------------------------------------------------------------------------

results = []

print("=" * 70)
print("American-Bermuda-Asian Option: LSM vs TVR-style Comparison")
print(f"K={K}, r={r}, T={T}, lockout={lockout}, n_paths={n_paths}")
print("=" * 70)
print()

for sigma in sigma_values:
    for S0 in S0_values:

        print(f"Pricing S0={S0:.0f}, sigma={sigma:.2f} ...", end="  ", flush=True)

        # European Asian MC (lower bound)
        euro_price, euro_se = price_european_asian_mc(
            S0, K, r, sigma, T=T, lookback=lockout,
            n_steps=n_steps, n_paths=n_paths, seed=seed
        )

        # LSM
        lsm_price, lsm_se = price_aba_lsmc(
            S0, K, r, sigma, T=T, lockout=lockout,
            n_steps=n_steps, n_paths=n_paths, seed=seed
        )

        # TVR-style
        tvr_price, tvr_se = price_aba_tvr(
            S0, K, r, sigma, T=T, lockout=lockout,
            n_steps=n_steps, n_paths=n_paths, seed=seed
        )

        # Early exercise premia
        lsm_eep = lsm_price - euro_price
        tvr_eep = tvr_price - euro_price
        lsm_vs_tvr = lsm_price - tvr_price

        print(f"Euro={euro_price:.4f}  LSM={lsm_price:.4f}  TVR={tvr_price:.4f}")

        results.append({
            "S0"            : S0,
            "K"             : K,
            "sigma"         : sigma,
            "r"             : r,
            "T"             : T,
            "european_asian": round(euro_price, 4),
            "european_se"   : round(euro_se,   4),
            "lsm_price"     : round(lsm_price,  4),
            "lsm_se"        : round(lsm_se,     4),
            "tvr_price"     : round(tvr_price,  4),
            "tvr_se"        : round(tvr_se,     4),
            "lsm_eep"       : round(lsm_eep,    4),   # early exercise premium
            "tvr_eep"       : round(tvr_eep,    4),
            "lsm_minus_tvr" : round(lsm_vs_tvr, 4),
        })

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

df = pd.DataFrame(results)

print()
print("=" * 70)
print("RESULTS TABLE")
print("=" * 70)
display_cols = ["S0", "european_asian", "lsm_price", "lsm_se",
                "tvr_price", "tvr_se", "lsm_eep", "tvr_eep", "lsm_minus_tvr"]
print(df[display_cols].to_string(index=False))
print()

# Save CSV
os.makedirs("data", exist_ok=True)
df.to_csv("data/asian_pricing_results.csv", index=False)
print("Saved: data/asian_pricing_results.csv")

# ---------------------------------------------------------------------------
# Figure: comparison bar chart
# ---------------------------------------------------------------------------

os.makedirs("visuals", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x      = np.arange(len(S0_values))
width  = 0.25
labels = [f"S0={s:.0f}" for s in S0_values]

# --- Left panel: absolute prices ---
ax = axes[0]
euro_vals = df["european_asian"].values
lsm_vals  = df["lsm_price"].values
tvr_vals  = df["tvr_price"].values

bars1 = ax.bar(x - width, euro_vals, width, label="European Asian (MC)",
               color="#4C72B0", alpha=0.85)
bars2 = ax.bar(x,          lsm_vals,  width, label="LSM (American-Bermuda-Asian)",
               color="#55A868", alpha=0.85)
bars3 = ax.bar(x + width,  tvr_vals,  width, label="TVR-style (American-Bermuda-Asian)",
               color="#C44E52", alpha=0.85)

# Error bars
ax.errorbar(x,         lsm_vals,  yerr=1.96*df["lsm_se"].values,
            fmt="none", color="black", capsize=4, linewidth=1.2)
ax.errorbar(x + width, tvr_vals,  yerr=1.96*df["tvr_se"].values,
            fmt="none", color="black", capsize=4, linewidth=1.2)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Option Price ($)")
ax.set_title("ABA Option Prices: LSM vs TVR-style\n(sigma=0.20, K=100, T=2)")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# --- Right panel: early exercise premium ---
ax2 = axes[1]
lsm_eep_vals = df["lsm_eep"].values
tvr_eep_vals = df["tvr_eep"].values

ax2.bar(x - width/2, lsm_eep_vals, width, label="LSM early exercise premium",
        color="#55A868", alpha=0.85)
ax2.bar(x + width/2, tvr_eep_vals, width, label="TVR early exercise premium",
        color="#C44E52", alpha=0.85)

ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_ylabel("Early Exercise Premium ($)")
ax2.set_title("Early Exercise Premium: LSM vs TVR-style\n(American price minus European Asian)")
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("visuals/asian_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: visuals/asian_model_comparison.png")

# ---------------------------------------------------------------------------
# LSM paper Table 3 reference (for manual comparison)
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("LSM PAPER TABLE 3 REFERENCE VALUES (sigma=0.20, K=100)")
print("(Early exercise value = American ABA - European Asian)")
print("=" * 70)
print("From Longstaff-Schwartz (2001) Table 3:")
print("  A=S=90,  sigma=0.20: early exercise ~ 0.04 - 0.10 range")
print("  A=S=100, sigma=0.20: early exercise ~ 0.18 - 0.25 range")
print("  A=S=110, sigma=0.20: early exercise ~ 0.50 - 0.80 range")
print("(Exact values depend on A0 vs S0 relationship in their setup)")
print()
print("Compare your LSM early exercise premium (lsm_eep) column above.")
print("If close, your LSM implementation is validated against the paper.")
