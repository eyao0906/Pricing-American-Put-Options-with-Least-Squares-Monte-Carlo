"""
asian_tvr.py
------------
TVR-style fitted value iteration for the American-Bermuda-Asian (ABA) option.

Same contract as asian_lsmc.py:
  - Call on arithmetic average A_t
  - Lockout at t = 0.25, expiry T = 2
  - Payoff: max(0, A_t - K)

Key differences from LSM:
  - Regresses on ALL paths (not ITM-only), consistent with TVR's
    global value-function approximation philosophy
  - Approximates V_t(S, A) directly at each step via fitted value iteration
  - Uses polynomial basis in both S and A (simpler than Laguerre)
  - Regression target is discounted NEXT-STEP VALUE ESTIMATE,
    not realized future cash flows (this is the TVR Q-function approach)

This makes the TVR/LSM comparison meaningful:
  - Same contract, same simulated paths, same basis dimensionality
  - Different regression target, different path selection, different architecture
"""

import numpy as np
from asian_lsmc import simulate_gbm_paths, compute_running_average


# ---------------------------------------------------------------------------
# Polynomial basis in two state variables (S, A)
# ---------------------------------------------------------------------------

def polynomial_basis_2d(S, A, K):
    """
    Build polynomial basis matrix for state (S, A).
    Basis: [1, s, s^2, a, a^2, s*a] where s=S/K, a=A/K.
    Returns design matrix of shape (n_paths, 6).

    This is the TVR-style global polynomial basis — same spirit as the
    [1, x, x^2, x^3] used in the vanilla TVR comparator but extended
    to the 2D state space.
    """
    s = S / K
    a = A / K

    X = np.column_stack([
        np.ones(len(S)),   # constant
        s,                 # S/K
        s**2,              # (S/K)^2
        a,                 # A/K
        a**2,              # (A/K)^2
        s * a,             # cross term
    ])
    return X


# ---------------------------------------------------------------------------
# TVR-style ABA pricing
# ---------------------------------------------------------------------------

def price_aba_tvr(S0, K, r, sigma,
                  T=2.0,
                  lockout=0.25,
                  n_steps=200,
                  n_paths=50000,
                  seed=42):
    """
    Price an American-Bermuda-Asian call using TVR-style fitted value iteration.

    TVR approach:
      At each time step t, approximate V_t(S, A) by:
        1. Take the current approximate V_{t+1} from the previous step
        2. Compute discounted next-step values: alpha * V_{t+1}(S_{t+1}, A_{t+1})
        3. Regress these on basis functions of (S_t, A_t) using ALL paths
        4. Set V_t = max(payoff_t, fitted_continuation_t)

    This is the Q-function / fitted value iteration loop from TVR Section III-D.
    The key theoretical point: using ALL paths (not ITM-only) means we are
    sampling from the trajectory distribution, which TVR Theorem 1 requires
    for bounded error. However, the polynomial basis may have larger epsilon
    (best approximation error) than Laguerre, causing error accumulation.

    Parameters
    ----------
    Same as price_aba_lsmc.

    Returns
    -------
    price : float - estimated option price
    se    : float - Monte Carlo standard error
    """

    dt = T / n_steps
    alpha = np.exp(-r * dt)

    lookback_steps = int(round(lockout / dt))
    lockout_step   = int(round(lockout / dt))

    # --- Simulate paths (same seed = same paths as LSM for fair comparison) ---
    S = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed)
    A = compute_running_average(S, dt, lookback_steps)

    # --- Payoff function ---
    def payoff(s, a):
        return np.maximum(0.0, a - K)

    # --- Initialise value estimates at maturity ---
    V = payoff(S[:, -1], A[:, -1])

    # --- Backward induction: TVR fitted value iteration ---
    for t in range(n_steps - 1, lockout_step - 1, -1):

        # Current state on all paths
        St = S[:, t]
        At = A[:, t]

        # Discounted next-step value estimate (TVR Q-function)
        Q = alpha * V   # this is the continuation value estimate

        # Build basis on ALL paths (TVR uses trajectory distribution, all paths)
        X = polynomial_basis_2d(St, At, K)

        # OLS regression: fit Q onto basis functions of current state
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, Q, rcond=None)
        except np.linalg.LinAlgError:
            # If regression fails, keep current V discounted
            V = Q
            continue

        # Fitted continuation value
        C_hat = X @ coeffs

        # Immediate exercise value
        exercise_val = payoff(St, At)

        # TVR value update: V_t = max(payoff, fitted continuation)
        V = np.maximum(exercise_val, C_hat)

        # Before lockout: can't exercise, so V = fitted continuation only
        # (lockout is handled by the loop range starting at lockout_step)

    # --- Price = discounted average value at lockout date ---
    # V now holds estimated value at t = lockout_step
    # Discount back to t = 0
    price_paths = np.exp(-r * lockout_step * dt) * V

    price = np.mean(price_paths)
    se    = np.std(price_paths) / np.sqrt(n_paths)

    return price, se


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    S0    = 100.0
    K     = 100.0
    r     = 0.06
    sigma = 0.20
    T     = 2.0
    lockout = 0.25

    print("=" * 55)
    print("American-Bermuda-Asian Option  (TVR-style)")
    print("Parameters: S0=100, K=100, r=0.06, sigma=0.20, T=2")
    print("=" * 55)

    tvr_price, tvr_se = price_aba_tvr(
        S0, K, r, sigma,
        T=T, lockout=lockout,
        n_steps=200, n_paths=50000, seed=42
    )

    print(f"TVR-style ABA price : {tvr_price:.4f}  (s.e. {tvr_se:.4f})")
