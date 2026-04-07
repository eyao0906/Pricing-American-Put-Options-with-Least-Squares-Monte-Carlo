"""
asian_lsmc.py
-------------
LSM pricing of an American-Bermuda-Asian (ABA) option.

Contract (following Longstaff-Schwartz Section 4 exactly):
  - Call option on the arithmetic average stock price A_t
  - Lockout period: not exercisable until t = 0.25
  - Final expiration: T = 2
  - Average A_t computed from t = -0.25 (3 months prior) to current time t
  - Payoff at exercise: max(0, A_t - K)

State variables at each exercise date:
  - S_t  : current stock price
  - A_t  : arithmetic average of S from lookback start to t

Basis functions (8 total, following LSM paper):
  - Constant
  - L1(S), L2(S)  : first two Laguerre polynomials in S/K
  - L1(A), L2(A)  : first two Laguerre polynomials in A/K
  - L1(S)*L1(A), L1(S)*L2(A), L2(S)*L1(A)  : cross products to 3rd order
"""

import numpy as np


# ---------------------------------------------------------------------------
# Laguerre polynomial basis (same as standard LSM practice)
# ---------------------------------------------------------------------------

def laguerre_basis_2d(S, A, K):
    """
    Build 8-column basis matrix for state (S, A).
    Inputs S and A are 1-D arrays of the same length (ITM paths only).
    Returns design matrix X of shape (n_paths, 8).
    """
    x = S / K
    a = A / K

    # Laguerre polynomials L1, L2 evaluated at x and a
    # L0 = exp(-x/2)  [absorbed into constant effectively]
    # L1 = exp(-x/2) * (1 - x)
    # L2 = exp(-x/2) * (1 - 2x + x^2/2)
    # We follow the standard LSM normalization without the exp weight
    # to keep things numerically stable.

    L1x = 1 - x
    L2x = 1 - 2*x + 0.5*x**2

    L1a = 1 - a
    L2a = 1 - 2*a + 0.5*a**2

    ones = np.ones(len(S))

    X = np.column_stack([
        ones,          # constant
        L1x,           # L1(S/K)
        L2x,           # L2(S/K)
        L1a,           # L1(A/K)
        L2a,           # L2(A/K)
        L1x * L1a,     # cross product
        L1x * L2a,     # cross product
        L2x * L1a,     # cross product
    ])
    return X


# ---------------------------------------------------------------------------
# GBM path simulation (same engine as your lsmc.py)
# ---------------------------------------------------------------------------

def simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=42):
    """
    Simulate risk-neutral GBM paths.
    Returns S of shape (n_paths, n_steps+1).
    Uses antithetic variates: first half normal, second half antithetic.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    half = n_paths // 2

    Z = rng.standard_normal((half, n_steps))
    Z = np.vstack([Z, -Z])           # antithetic

    log_increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    for t in range(1, n_steps + 1):
        S[:, t] = S[:, t-1] * np.exp(log_increments[:, t-1])

    return S


# ---------------------------------------------------------------------------
# Running average computation
# ---------------------------------------------------------------------------

def compute_running_average(S, dt, lookback_steps):
    """
    Compute arithmetic average A_t for each path and each time step.

    The average starts from t = -lookback (3 months prior).
    We approximate: at time index i (corresponding to real time t_i),
    the average A_{t_i} = mean of S from lookback_start to t_i.

    Since we don't simulate before t=0, we treat S_0 as the price
    throughout the lookback window (a standard simplification when
    no historical path is given). This matches common practice.

    Returns A of shape (n_paths, n_steps+1).
    """
    n_paths, n_cols = S.shape
    n_steps = n_cols - 1
    A = np.zeros_like(S)

    # At t=0: average over lookback window where price = S0 throughout
    # so A_0 = S_0
    A[:, 0] = S[:, 0]

    # Cumulative sum approach for efficiency
    # At step i, the average covers lookback_steps + i + 1 observations
    # (lookback window before t=0, plus steps 0..i)
    cumsum = S[:, 0].copy()  # running sum from step 0 onward

    for i in range(1, n_steps + 1):
        cumsum += S[:, i]
        total_obs = lookback_steps + i + 1  # lookback + steps 0..i
        # Lookback contribution: lookback_steps * S_0
        lookback_contrib = lookback_steps * S[:, 0]
        A[:, i] = (lookback_contrib + cumsum) / total_obs

    return A


# ---------------------------------------------------------------------------
# Main LSM pricing function for the ABA option
# ---------------------------------------------------------------------------

def price_aba_lsmc(S0, K, r, sigma,
                   T=2.0,
                   lockout=0.25,
                   n_steps=200,
                   n_paths=50000,
                   seed=42):
    """
    Price an American-Bermuda-Asian call option using LSM.

    Parameters
    ----------
    S0      : float  - initial stock price
    K       : float  - strike price
    r       : float  - risk-free rate (annual)
    sigma   : float  - volatility (annual)
    T       : float  - maturity in years (default 2.0, per LSM paper)
    lockout : float  - earliest exercise time (default 0.25, per LSM paper)
    n_steps : int    - number of time steps (100 per year -> 200 for T=2)
    n_paths : int    - number of paths (use antithetic, so half unique)
    seed    : int    - random seed

    Returns
    -------
    price   : float  - estimated option price
    se      : float  - Monte Carlo standard error
    """

    dt = T / n_steps
    discount = np.exp(-r * dt)

    # Lookback: 3 months = 0.25 years before t=0
    lookback_steps = int(round(lockout / dt))  # number of steps in lookback

    # Lockout: first exercise date at t = lockout
    lockout_step = int(round(lockout / dt))

    # --- Simulate paths ---
    S = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed)

    # --- Compute running averages ---
    A = compute_running_average(S, dt, lookback_steps)

    # --- Payoff function ---
    def payoff(s, a):
        return np.maximum(0.0, a - K)

    # --- Initialise cash flow matrix ---
    # cash_flow[i] = discounted cash flow received on path i
    # Start at maturity
    cash_flow = payoff(S[:, -1], A[:, -1])

    # --- Backward induction ---
    for t in range(n_steps - 1, lockout_step - 1, -1):

        # Discount cash flows one step
        cash_flow *= discount

        # Current state
        St = S[:, t]
        At = A[:, t]

        # Immediate exercise value
        exercise_val = payoff(St, At)

        # Only consider ITM paths for regression
        itm = exercise_val > 0
        if itm.sum() < 10:
            # Too few ITM paths, skip regression at this step
            continue

        St_itm = St[itm]
        At_itm = At[itm]
        Y = cash_flow[itm]   # discounted future cash flows on ITM paths

        # Build basis matrix
        X = laguerre_basis_2d(St_itm, At_itm, K)

        # OLS regression: continuation value estimate
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        continuation = X @ coeffs

        # Exercise decision on ITM paths
        exercise_now = exercise_val[itm] > continuation

        # Update cash flows: if exercising now, replace with immediate payoff
        # (undiscounted relative to current time, since we already discounted above)
        # We need to undo the one-step discount for paths that exercise now
        # and replace with the immediate payoff
        updated = cash_flow[itm].copy()
        updated[exercise_now] = exercise_val[itm][exercise_now] / discount  # undo discount
        cash_flow[itm] = updated

        # For paths where we exercise now, lock in that cash flow
        # (they won't be reconsidered at earlier steps — handled naturally
        #  since we replace their value)

    # Final discount back to t=0
    # At this point cash_flow holds the cash flow received at the exercise
    # date for each path, already discounted to t=lockout_step level.
    # We need to discount from lockout_step back to t=0.
    cash_flow *= np.exp(-r * lockout_step * dt)

    price = np.mean(cash_flow)
    se = np.std(cash_flow) / np.sqrt(n_paths)

    return price, se


# ---------------------------------------------------------------------------
# European Asian call benchmark (closed-form approximation via Turnbull-Wakeman)
# For comparison purposes - gives a lower bound since American >= European
# ---------------------------------------------------------------------------

def price_european_asian_mc(S0, K, r, sigma,
                             T=2.0,
                             lookback=0.25,
                             n_steps=200,
                             n_paths=50000,
                             seed=42):
    """
    Price the European counterpart of the ABA option by Monte Carlo.
    Payoff = max(0, A_T - K) where A_T is the average from -lookback to T.
    Exercise only at T.
    """
    dt = T / n_steps
    lookback_steps = int(round(lookback / dt))

    S = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed)
    A = compute_running_average(S, dt, lookback_steps)

    # Terminal average payoff
    payoff = np.maximum(0.0, A[:, -1] - K)
    discounted = np.exp(-r * T) * payoff

    price = np.mean(discounted)
    se = np.std(discounted) / np.sqrt(n_paths)
    return price, se


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Parameters from LSM paper Section 4
    S0    = 100.0
    K     = 100.0
    r     = 0.06
    sigma = 0.20
    T     = 2.0
    lockout = 0.25

    print("=" * 55)
    print("American-Bermuda-Asian Option  (LSM)")
    print("Parameters: S0=100, K=100, r=0.06, sigma=0.20, T=2")
    print("=" * 55)

    aba_price, aba_se = price_aba_lsmc(
        S0, K, r, sigma,
        T=T, lockout=lockout,
        n_steps=200, n_paths=50000, seed=42
    )

    euro_price, euro_se = price_european_asian_mc(
        S0, K, r, sigma,
        T=T, lookback=lockout,
        n_steps=200, n_paths=50000, seed=42
    )

    early_exercise = aba_price - euro_price

    print(f"American-Bermuda-Asian price : {aba_price:.4f}  (s.e. {aba_se:.4f})")
    print(f"European Asian price         : {euro_price:.4f}  (s.e. {euro_se:.4f})")
    print(f"Early exercise premium       : {early_exercise:.4f}")
    print()
    print("LSM paper Table 3 reference values (A=S=100, sigma=0.20):")
    print("  American early exercise value ~ 0.18 to 0.25 range")
    print("  (exact depends on A0 relative to S0)")
