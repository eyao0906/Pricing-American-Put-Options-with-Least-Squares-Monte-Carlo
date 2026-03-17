import numpy as np
from dataclasses import dataclass


@dataclass
class LSMCResult:
    price: float
    stderr: float
    exercise_boundary: np.ndarray


def laguerre_basis(x: np.ndarray, degree: int = 3) -> np.ndarray:
    """Simple Laguerre polynomial basis used by Longstaff-Schwartz (2001)."""
    x = np.asarray(x)
    cols = [np.ones_like(x)]
    if degree >= 1:
        cols.append(1.0 - x)
    if degree >= 2:
        cols.append(1.0 - 2.0 * x + 0.5 * x**2)
    if degree >= 3:
        cols.append(1.0 - 3.0 * x + 1.5 * x**2 - (x**3) / 6.0)
    if degree > 3:
        for d in range(4, degree + 1):
            cols.append(x**d)
    return np.column_stack(cols)


def simulate_gbm_paths(
    s0: float,
    r: float,
    sigma: float,
    t: float,
    steps: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    dt = t / steps
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, steps))
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.column_stack([np.zeros(n_paths), log_paths])
    return s0 * np.exp(log_paths)


def price_american_put_lsmc(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float = 1.0,
    steps: int = 50,
    n_paths: int = 100_000,
    basis_degree: int = 3,
    seed: int = 42,
) -> LSMCResult:
    paths = simulate_gbm_paths(s0, r, sigma, t, steps, n_paths, seed)
    dt = t / steps
    discount = np.exp(-r * dt)

    intrinsic = np.maximum(k - paths, 0.0)
    cf = intrinsic[:, -1].copy()
    exercise_boundary = np.full(steps + 1, np.nan)
    exercise_boundary[-1] = k

    for t_idx in range(steps - 1, 0, -1):
        itm = intrinsic[:, t_idx] > 0
        if np.any(itm):
            x = paths[itm, t_idx] / k
            y = cf[itm] * discount
            xmat = laguerre_basis(x, degree=basis_degree)
            beta, *_ = np.linalg.lstsq(xmat, y, rcond=None)
            continuation = xmat @ beta

            exercise = intrinsic[itm, t_idx] > continuation
            itm_idx = np.where(itm)[0]
            ex_idx = itm_idx[exercise]
            cont_idx = itm_idx[~exercise]

            cf[ex_idx] = intrinsic[ex_idx, t_idx]
            cf[cont_idx] = cf[cont_idx] * discount

            if len(ex_idx) > 0:
                exercise_boundary[t_idx] = np.max(paths[ex_idx, t_idx])
        non_itm = ~itm
        cf[non_itm] *= discount

    pv = cf * discount
    price = np.mean(pv)
    stderr = np.std(pv, ddof=1) / np.sqrt(n_paths)
    return LSMCResult(price=price, stderr=stderr, exercise_boundary=exercise_boundary)
