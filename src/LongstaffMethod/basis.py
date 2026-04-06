from __future__ import annotations
from typing import Iterable
import numpy as np

Array = np.ndarray

def weighted_laguerre_polynomials(x: Array, order: int = 4) -> Array:
    """Return weighted Laguerre basis columns e^{-x/2} L_n(x), n=0,...,order-1.

    For the vanilla benchmark we use order=4, i.e. constant plus the first three
    weighted Laguerre polynomials, matching the usual Longstaff-Schwartz style.
    """
    x = np.asarray(x, dtype=float)
    if order < 1:
        raise ValueError("order must be at least 1")

    w = np.exp(-0.5 * x)
    cols: list[Array] = [w]
    if order >= 2:
        cols.append(w * (1.0 - x))
    if order >= 3:
        cols.append(w * (1.0 - 2.0 * x + 0.5 * x**2))
    if order >= 4:
        cols.append(w * (1.0 - 3.0 * x + 1.5 * x**2 - (x**3) / 6.0))
    if order >= 5:
        cols.append(w * (1.0 - 4.0 * x + 3.0 * x**2 - (2.0 / 3.0) * x**3 + x**4 / 24.0))
    if order > 5:
        for power in range(5, order):
            cols.append(w * x**power)
    return np.column_stack(cols)


def vanilla_put_basis(spot: Array, strike: float, basis_size: int = 4) -> Array:
    x = np.asarray(spot, dtype=float) / strike
    return weighted_laguerre_polynomials(x, order=basis_size)


def asian_state_basis(spot: Array, average: Array, strike: float) -> Array:
    """Compact 2D basis for Example 2 with spot/average cross terms."""
    xs = np.asarray(spot, dtype=float) / strike
    xa = np.asarray(average, dtype=float) / strike
    ls = weighted_laguerre_polynomials(xs, order=3)
    la = weighted_laguerre_polynomials(xa, order=3)
    cols: list[Array] = [
        np.ones_like(xs),
        ls[:, 0],
        ls[:, 1],
        la[:, 0],
        la[:, 1],
        xs * xa,
        xs,
        xa,
        ls[:, 1] * la[:, 1],
    ]
    return np.column_stack(cols)


def select_representative_rows(values: Iterable[float], max_rows: int = 5) -> list[float]:
    vals = list(values)
    if len(vals) <= max_rows:
        return vals
    idx = np.linspace(0, len(vals) - 1, num=max_rows).round().astype(int)
    return [vals[i] for i in idx]
