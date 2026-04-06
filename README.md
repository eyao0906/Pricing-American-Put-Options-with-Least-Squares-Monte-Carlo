# ACTSC 971 American Option Pricing Project

This refactor centers the project on a synthetic Longstaff–Schwartz style benchmark instead of historical AAPL calibration.

## What is implemented

Under `src/`:

- `basis.py`
  - weighted Laguerre basis for 1D vanilla put regression
  - 2D basis with cross terms for the Asian example
- `simulators.py`
  - GBM with antithetic variates
  - GBM with running arithmetic average
  - jump-to-ruin simulator
- `lsm_engine.py`
  - reusable Longstaff–Schwartz backward induction engine
  - fitted stopping-rule object for out-of-sample diagnostics
- `benchmarks.py`
  - Black–Scholes European put
  - implicit finite-difference American put benchmark
  - CRR tree sanity benchmark
- `run_longstaff_vanilla.py`
  - paper-style synthetic benchmark and robustness studies
- `run_longstaff_asian.py`
  - Example 2: American–Bermuda–Asian option with multiple state variables
- `run_longstaff_jump.py`
  - Example 4: jump-diffusion / jump-to-ruin American put example
- `run_study.py`
  - convenience entry point to the vanilla benchmark workflow

## Main vanilla benchmark setup

- Strike `K = 40`
- Risk-free rate `r = 0.06`
- Spots `S in {36, 38, 40, 42, 44}`
- Volatility `sigma in {0.20, 0.40}`
- Maturity `T in {1, 2}`
- Exercise dates: `50` per year
- Paths: `100,000 total = 50,000 base + 50,000 antithetic`
- Basis: constant + first three weighted Laguerre polynomials

## Outputs

Tables are written to `outputs/`, visuals to `visuals/`, and configuration snapshots to `data/`.

## Run commands

From the project root:

```bash
python src/run_longstaff_vanilla.py
python src/run_longstaff_asian.py
python src/run_longstaff_jump.py
```

Or equivalently for the main benchmark:

```bash
python src/run_study.py
```

## Notes

- The vanilla American put uses implicit finite difference as the primary direct benchmark.
- The Asian example is intentionally compact and does not include a direct PDE benchmark.
- The jump example reuses the same LSM engine and changes only the simulator/model side.
