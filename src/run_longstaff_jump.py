from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from basis import vanilla_put_basis
from benchmarks import black_scholes_put
from lsm_engine import fit_lsm
from simulators import simulate_gbm_paths, simulate_jump_to_ruin_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
VISUALS_DIR = PROJECT_ROOT / "visuals"


def put_payoff(states: dict[str, np.ndarray], t: int, strike: float) -> np.ndarray:
    return np.maximum(strike - states["spot"][:, t], 0.0)


def always_exercisable(states: dict[str, np.ndarray], t: int) -> np.ndarray:
    return np.ones(states["spot"].shape[0], dtype=bool)


def price_case(simulator_name: str, sim, strike: float, r: float, maturity: float) -> tuple[float, float, list[dict[str, float]]]:
    payoff_fn = lambda states, t: put_payoff(states, t, strike=strike)
    basis_fn = lambda states, t: vanilla_put_basis(states["spot"][:, t], strike=strike, basis_size=4)
    result, _ = fit_lsm(
        states=sim.states,
        payoff_fn=payoff_fn,
        basis_fn=basis_fn,
        exercise_mask_fn=always_exercisable,
        r=r,
        maturity=maturity,
    )
    return result.price, result.stderr, result.exercise_summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    strike = 40.0
    s0 = 40.0
    r = 0.06
    sigma = 0.20
    maturity = 1.0
    steps = 50
    n_paths = 100_000
    seed = 20260426
    jump_intensity = 0.40

    no_jump_sim = simulate_gbm_paths(
        s0=s0,
        r=r,
        sigma=sigma,
        maturity=maturity,
        steps=steps,
        n_paths=n_paths,
        seed=seed,
        antithetic=True,
    )
    jump_sim = simulate_jump_to_ruin_paths(
        s0=s0,
        r=r,
        sigma=sigma,
        maturity=maturity,
        steps=steps,
        n_paths=n_paths,
        jump_intensity=jump_intensity,
        seed=seed,
        antithetic=True,
    )

    no_jump_price, no_jump_stderr, no_jump_boundary = price_case("no_jump", no_jump_sim, strike, r, maturity)
    jump_price, jump_stderr, jump_boundary = price_case("jump_to_ruin", jump_sim, strike, r, maturity)

    euro_no_jump = black_scholes_put(s0, strike, r, sigma, maturity)
    euro_jump_mc = float(np.exp(-r * maturity) * np.maximum(strike - jump_sim.states["spot"][:, -1], 0.0).mean())

    results = pd.DataFrame(
        [
            {
                "model": "no_jump_gbm",
                "spot0": s0,
                "strike": strike,
                "sigma": sigma,
                "risk_free": r,
                "maturity": maturity,
                "jump_intensity": 0.0,
                "american_lsm": no_jump_price,
                "lsm_stderr": no_jump_stderr,
                "european_value": euro_no_jump,
                "early_exercise_premium": no_jump_price - euro_no_jump,
            },
            {
                "model": "jump_to_ruin",
                "spot0": s0,
                "strike": strike,
                "sigma": sigma,
                "risk_free": r,
                "maturity": maturity,
                "jump_intensity": jump_intensity,
                "american_lsm": jump_price,
                "lsm_stderr": jump_stderr,
                "european_value": euro_jump_mc,
                "early_exercise_premium": jump_price - euro_jump_mc,
            },
        ]
    )
    results.to_csv(OUTPUT_DIR / "longstaff_jump_results.csv", index=False)

    plt.figure(figsize=(8, 5))
    for boundary, label in [(no_jump_boundary, "No jump"), (jump_boundary, "Jump-to-ruin")]:
        if boundary:
            df = pd.DataFrame(boundary)
            plt.plot(df["time"], df["boundary_mean"], marker="o", label=label)
    plt.xlabel("Time")
    plt.ylabel("Mean exercise spot among exercised paths")
    plt.title("Jump example: empirical exercise boundary comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "longstaff_jump_boundary_comparison.png", dpi=180)
    plt.close()

    print(results.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
