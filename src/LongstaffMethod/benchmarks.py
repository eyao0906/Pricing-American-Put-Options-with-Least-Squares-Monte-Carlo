from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def black_scholes_put(spot: float, strike: float, r: float, sigma: float, maturity: float) -> float:
    if maturity <= 0:
        return max(strike - spot, 0.0)
    if sigma <= 0:
        return max(strike * math.exp(-r * maturity) - spot, 0.0)
    sqrt_t = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma**2) * maturity) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return float(strike * math.exp(-r * maturity) * norm.cdf(-d2) - spot * norm.cdf(-d1))


def american_put_binomial_crr(
    spot: float,
    strike: float,
    r: float,
    sigma: float,
    maturity: float,
    steps: int = 1000,
) -> float:
    dt = maturity / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    j = np.arange(steps + 1)
    stock = spot * (u ** j) * (d ** (steps - j))
    value = np.maximum(strike - stock, 0.0)
    for n in range(steps - 1, -1, -1):
        value = disc * (p * value[1:] + (1.0 - p) * value[:-1])
        j = np.arange(n + 1)
        stock = spot * (u ** j) * (d ** (n - j))
        value = np.maximum(value, np.maximum(strike - stock, 0.0))
    return float(value[0])


def american_put_implicit_fd(
    spot: float,
    strike: float,
    r: float,
    sigma: float,
    maturity: float,
    exercise_steps: int,
    s_max_multiplier: float = 4.0,
    s_nodes: int = 400,
) -> float:
    """Implicit finite difference for the Black-Scholes American put.

    Uses a tridiagonal solve on a uniform S-grid and projects onto intrinsic value.
    This is the primary ground-truth benchmark for the vanilla synthetic study.
    """
    s_max = max(s_max_multiplier * strike, s_max_multiplier * spot, strike * 2.5)
    ds = s_max / s_nodes
    dt = maturity / exercise_steps
    grid = np.linspace(0.0, s_max, s_nodes + 1)
    values = np.maximum(strike - grid, 0.0)

    i = np.arange(1, s_nodes)
    alpha = 0.5 * dt * (sigma**2 * i**2 - r * i)
    beta = dt * (sigma**2 * i**2 + r)
    gamma = 0.5 * dt * (sigma**2 * i**2 + r * i)

    lower = -alpha[1:]
    diag = 1.0 + beta
    upper = -gamma[:-1]

    for n in range(exercise_steps - 1, -1, -1):
        tau = maturity - n * dt
        rhs = values[1:-1].copy()
        left_boundary = strike
        right_boundary = 0.0
        rhs[0] += alpha[0] * left_boundary
        rhs[-1] += gamma[-1] * right_boundary

        c_prime = np.empty_like(upper)
        d_prime = np.empty_like(rhs)
        c_prime[0] = upper[0] / diag[0]
        d_prime[0] = rhs[0] / diag[0]
        for k in range(1, len(rhs) - 1):
            denom = diag[k] - lower[k - 1] * c_prime[k - 1]
            c_prime[k] = upper[k] / denom if k < len(upper) else 0.0
            d_prime[k] = (rhs[k] - lower[k - 1] * d_prime[k - 1]) / denom
        d_prime[-1] = (rhs[-1] - lower[-1] * d_prime[-2]) / (diag[-1] - lower[-1] * c_prime[-2])

        interior = np.empty_like(rhs)
        interior[-1] = d_prime[-1]
        for k in range(len(rhs) - 2, -1, -1):
            interior[k] = d_prime[k] - c_prime[k] * interior[k + 1]

        values[0] = left_boundary
        values[-1] = right_boundary
        values[1:-1] = np.maximum(interior, strike - grid[1:-1])

    return float(np.interp(spot, grid, values))
