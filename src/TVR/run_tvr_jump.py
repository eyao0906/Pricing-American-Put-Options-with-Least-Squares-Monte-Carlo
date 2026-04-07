from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LONGSTAFF_DIR = PROJECT_ROOT / "src" / "LongstaffMethod"
if str(LONGSTAFF_DIR) not in sys.path:
    sys.path.append(str(LONGSTAFF_DIR))

from basis import vanilla_put_basis
from benchmarks import black_scholes_put
from simulators import simulate_gbm_paths, simulate_jump_to_ruin_paths
from tvr_engine import fit_tvr


OUTPUT_DIR = PROJECT_ROOT / "outputs"
VISUALS_DIR = PROJECT_ROOT / "visuals"


def put_payoff(states: dict[str, np.ndarray], t: int, strike: float) -> np.ndarray:
    return np.maximum(strike - states["spot"][:, t], 0.0)


def always_exercisable(states: dict[str, np.ndarray], t: int) -> np.ndarray:
    return np.ones(states["spot"].shape[0], dtype=bool)


def fit_and_eval(sim_fit, sim_eval, strike: float, r: float, maturity: float) -> tuple[float, float, float, list[dict[str, float]]]:
    payoff_fn = lambda states, t: put_payoff(states, t, strike)
    basis_fn = lambda states, t: vanilla_put_basis(states["spot"][:, t], strike=strike, basis_size=4)
    in_result, policy = fit_tvr(
        states=sim_fit.states,
        payoff_fn=payoff_fn,
        basis_fn=basis_fn,
        exercise_mask_fn=always_exercisable,
        r=r,
        maturity=maturity,
    )
    out_price, out_stderr, stopping_times, _ = policy.evaluate(sim_eval.states, r)
    steps = sim_eval.states["spot"].shape[1] - 1
    dt = maturity / steps
    boundary = []
    for t in range(1, steps):
        idx = np.where(stopping_times == t)[0]
        if idx.size > 0:
            spot = sim_eval.states["spot"][idx, t]
            boundary.append({"time": t * dt, "boundary_mean": float(np.mean(spot))})
    return in_result.price, out_price, out_stderr, boundary


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
    seed = 20260427
    jump_intensity = 0.40

    fit_no_jump = simulate_gbm_paths(s0, r, sigma, maturity, steps, n_paths, seed=seed, antithetic=True)
    eval_no_jump = simulate_gbm_paths(s0, r, sigma, maturity, steps, n_paths, seed=seed + 7919, antithetic=True)
    fit_jump = simulate_jump_to_ruin_paths(s0, r, sigma, maturity, steps, n_paths, jump_intensity, seed=seed, antithetic=True)
    eval_jump = simulate_jump_to_ruin_paths(s0, r, sigma, maturity, steps, n_paths, jump_intensity, seed=seed + 7919, antithetic=True)

    no_jump_in, no_jump_out, no_jump_stderr, no_jump_boundary = fit_and_eval(fit_no_jump, eval_no_jump, strike, r, maturity)
    jump_in, jump_out, jump_stderr, jump_boundary = fit_and_eval(fit_jump, eval_jump, strike, r, maturity)

    euro_no_jump = black_scholes_put(s0, strike, r, sigma, maturity)
    euro_jump_mc = float(np.exp(-r * maturity) * np.maximum(strike - eval_jump.states["spot"][:, -1], 0.0).mean())

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
                "tvr_price_in_sample": no_jump_in,
                "tvr_price_out_of_sample": no_jump_out,
                "tvr_stderr_out_of_sample": no_jump_stderr,
                "european_value": euro_no_jump,
                "early_exercise_premium": no_jump_out - euro_no_jump,
            },
            {
                "model": "jump_to_ruin",
                "spot0": s0,
                "strike": strike,
                "sigma": sigma,
                "risk_free": r,
                "maturity": maturity,
                "jump_intensity": jump_intensity,
                "tvr_price_in_sample": jump_in,
                "tvr_price_out_of_sample": jump_out,
                "tvr_stderr_out_of_sample": jump_stderr,
                "european_value": euro_jump_mc,
                "early_exercise_premium": jump_out - euro_jump_mc,
            },
        ]
    )
    results.to_csv(OUTPUT_DIR / "tvr_jump_results.csv", index=False)

    plt.figure(figsize=(8, 5))
    for boundary, label in [(no_jump_boundary, "TVR no jump"), (jump_boundary, "TVR jump-to-ruin")]:
        if boundary:
            df = pd.DataFrame(boundary)
            plt.plot(df["time"], df["boundary_mean"], marker="o", label=label)
    plt.xlabel("Time")
    plt.ylabel("Mean exercise spot among exercised paths")
    plt.title("TVR jump study: empirical boundary comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "tvr_jump_boundary_comparison.png", dpi=180)
    plt.close()

    print(results.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
