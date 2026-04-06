from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass
class SimulationResult:
    times: Array
    primary_state: Array
    states: dict[str, Array]
    metadata: dict[str, float | int | str]


def _validate_even_paths(n_paths: int, antithetic: bool) -> tuple[int, int]:
    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("n_paths must be even when antithetic=True")
        return n_paths // 2, n_paths
    return n_paths, n_paths


def simulate_gbm_paths(
    s0: float,
    r: float,
    sigma: float,
    maturity: float,
    steps: int,
    n_paths: int,
    seed: int = 0,
    antithetic: bool = True,
) -> SimulationResult:
    """Risk-neutral GBM with optional antithetic Brownian shocks."""
    base_paths, total_paths = _validate_even_paths(n_paths, antithetic)
    dt = maturity / steps
    times = np.linspace(0.0, maturity, steps + 1)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((base_paths, steps))
    z_full = np.vstack([z, -z]) if antithetic else z

    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * z_full
    log_increments = drift + diffusion
    log_paths = np.cumsum(log_increments, axis=1)
    log_paths = np.column_stack([np.zeros(total_paths), log_paths])
    spot = s0 * np.exp(log_paths)

    return SimulationResult(
        times=times,
        primary_state=spot,
        states={"spot": spot},
        metadata={
            "s0": s0,
            "r": r,
            "sigma": sigma,
            "maturity": maturity,
            "steps": steps,
            "n_paths": total_paths,
            "antithetic": int(antithetic),
            "seed": seed,
        },
    )


def simulate_gbm_with_running_average(
    s0: float,
    r: float,
    sigma: float,
    maturity: float,
    steps: int,
    n_paths: int,
    seed: int = 0,
    antithetic: bool = True,
) -> SimulationResult:
    sim = simulate_gbm_paths(
        s0=s0,
        r=r,
        sigma=sigma,
        maturity=maturity,
        steps=steps,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )
    spot = sim.states["spot"]
    counts = np.arange(1, spot.shape[1] + 1)
    running_average = np.cumsum(spot, axis=1) / counts[None, :]
    sim.states["average"] = running_average
    return sim


def simulate_jump_to_ruin_paths(
    s0: float,
    r: float,
    sigma: float,
    maturity: float,
    steps: int,
    n_paths: int,
    jump_intensity: float,
    seed: int = 0,
    antithetic: bool = True,
) -> SimulationResult:
    """GBM diffusion plus Poisson jump-to-ruin. Once ruin occurs, S stays at 0."""
    base_paths, total_paths = _validate_even_paths(n_paths, antithetic)
    dt = maturity / steps
    times = np.linspace(0.0, maturity, steps + 1)
    rng = np.random.default_rng(seed)

    z = rng.standard_normal((base_paths, steps))
    z_full = np.vstack([z, -z]) if antithetic else z

    jump_uniforms = rng.random((total_paths, steps))
    jump_occurs = jump_uniforms < 1.0 - np.exp(-jump_intensity * dt)

    spot = np.empty((total_paths, steps + 1), dtype=float)
    ruined = np.zeros(total_paths, dtype=bool)
    spot[:, 0] = s0
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    for t in range(steps):
        alive = ~ruined
        next_values = np.zeros(total_paths, dtype=float)
        if np.any(alive):
            next_values[alive] = spot[alive, t] * np.exp(drift + vol * z_full[alive, t])
        newly_ruined = alive & jump_occurs[:, t]
        next_values[newly_ruined] = 0.0
        ruined = ruined | newly_ruined
        spot[:, t + 1] = next_values

    return SimulationResult(
        times=times,
        primary_state=spot,
        states={"spot": spot},
        metadata={
            "s0": s0,
            "r": r,
            "sigma": sigma,
            "maturity": maturity,
            "steps": steps,
            "n_paths": total_paths,
            "jump_intensity": jump_intensity,
            "antithetic": int(antithetic),
            "seed": seed,
        },
    )
