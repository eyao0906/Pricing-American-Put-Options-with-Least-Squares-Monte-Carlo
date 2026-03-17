# American Option Pricing Study (LSMC + Extension)

This project studies American put option pricing using:

1. **Longstaff-Schwartz Least Squares Monte Carlo (LSMC)**
2. **Tsitsiklis-Van Roy style regression dynamic programming (extension/comparison)**

Primary reference:
- Longstaff & Schwartz (2001), *Valuing American Options by Simulation: A Simple Least-Squares Approach*
- URL: <https://galton.uchicago.edu/~mykland/346W07/Longstaff.pdf>

Related/extended reference:
- Tsitsiklis & Van Roy (2001), *Regression Methods for Pricing Complex American-Style Options*
- URL: <https://www.mit.edu/~jnt/Papers/J086-01-bvr-options.pdf>

## Project architecture

```text
.
├─ data/
│  ├─ aapl_prices.csv
│  └─ pricing_results.csv
├─ src/
│  ├─ data_loader.py
│  ├─ lsmc.py
│  ├─ tsitsiklis_vanroy.py
│  └─ run_study.py
├─ visuals/
│  ├─ historical_price.png
│  ├─ exercise_boundary.png
│  └─ model_comparison.png
└─ report/
   └─ american_options_lsmc_study.qmd
```

## What the code does

- Downloads historical price data from Yahoo Finance (`AAPL`) via `yfinance`
- Estimates annualized volatility from 5Y daily log returns
- Uses `^IRX` as a simple risk-free proxy
- Simulates GBM paths and prices American puts for multiple strikes
- Produces:
  - pricing comparison table (LSMC vs TVR + Black-Scholes European reference)
  - exercise-boundary visualization
  - report in Quarto (`.qmd`, renderable to PDF)

## Run instructions

```bash
python -m pip install yfinance pandas numpy matplotlib
python src/run_study.py
```

Render report (PDF):

```bash
quarto render report/american_options_lsmc_study.qmd --to pdf
```

Fallback (HTML):

```bash
quarto render report/american_options_lsmc_study.qmd --to html
```

## Notes / assumptions

- Underlying dynamics in this implementation are GBM with constant volatility/rate.
- This is a research/educational replication, not production pricing code.
- TVR implementation is a compact fitted-value-iteration style comparator.
