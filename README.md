# American Option Pricing Project

Under `src/LongstaffMethod`:

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

Under `src/TVR`:

- `basis.py`
  - same module as Longstaff
- `simulators.py`
  - same module as Longstaff
- `TVR_engine.py`
  - TVR regression engine
  - fitted stopping-rule object for out-of-sample diagnostics
- `benchmarks.py`
  - same module as Longstaff
- `run_tvr_vanilla.py`
  - paper-style synthetic benchmark and robustness studies
- `run_tvr_asian.py`
  - Example 2: American–Bermuda–Asian option with multiple state variables
- `run_tvr_jump.py`
  - Example 4: jump-diffusion / jump-to-ruin American put example

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
python src/LongstaffMethod/run_longstaff_vanilla.py
python src/LongstaffMethod/run_longstaff_asian.py
python src/LongstaffMethod/run_longstaff_jump.py

python src/TVR/run_longstaff_vanilla.py
python src/TVR/run_longstaff_asian.py
python src/TVR/run_longstaff_jump.py
```

## Notes

- The vanilla American put uses implicit finite difference as the primary direct benchmark.
- The Asian example is intentionally compact and does not include a direct PDE benchmark.
- The jump example reuses the same LSM/TVR engine and changes only the simulator/model side.
