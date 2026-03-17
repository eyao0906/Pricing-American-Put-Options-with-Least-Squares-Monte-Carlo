import numpy as np
from dataclasses import dataclass

from lsmc import simulate_gbm_paths


@dataclass
class TVRResult:
    price: float
    stderr: float


def polynomial_basis(s: np.ndarray, k: float) -> np.ndarray:
    x = s / k
    return np.column_stack([np.ones_like(x), x, x**2, x**3])


def price_american_put_tvr(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float = 1.0,
    steps: int = 50,
    n_paths: int = 100_000,
    seed: int = 123,
) -> TVRResult:
    """Tsitsiklis-Van Roy style fitted value iteration (2001) with regression on all paths."""
    paths = simulate_gbm_paths(s0, r, sigma, t, steps, n_paths, seed)
    dt = t / steps
    discount = np.exp(-r * dt)

    value_next = np.maximum(k - paths[:, -1], 0.0)

    for t_idx in range(steps - 1, -1, -1):
        s_t = paths[:, t_idx]
        intrinsic = np.maximum(k - s_t, 0.0)

        x = polynomial_basis(s_t, k)
        y = discount * value_next
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        continuation = x @ beta

        value_now = np.maximum(intrinsic, continuation)
        value_next = value_now

    # At t=0 all paths share the same state S0, so the fitted value is effectively deterministic.
    price = float(np.mean(value_next))
    stderr = float("nan")
    return TVRResult(price=price, stderr=stderr)
