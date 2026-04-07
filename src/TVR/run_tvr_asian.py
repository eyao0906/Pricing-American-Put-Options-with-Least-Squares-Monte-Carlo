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

from basis import asian_state_basis
from simulators import simulate_gbm_with_running_average
from tvr_engine import fit_tvr


OUTPUT_DIR = PROJECT_ROOT / "outputs"
VISUALS_DIR = PROJECT_ROOT / "visuals"


def asian_payoff(states: dict[str, np.ndarray], t: int, strike: float) -> np.ndarray:
    return np.maximum(strike - states["average"][:, t], 0.0)


def lockout_mask(states: dict[str, np.ndarray], t: int, lockout_time: float, maturity: float) -> np.ndarray:
    steps = states["spot"].shape[1] - 1
    time = maturity * t / steps
    return np.full(states["spot"].shape[0], time >= lockout_time, dtype=bool)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    strike = 40.0
    s0 = 40.0
    r = 0.06
    sigma = 0.20
    maturity = 2.0
    lockout_time = 0.25
    steps = 100
    n_paths = 100_000
    seed = 20260417

    fit_sim = simulate_gbm_with_running_average(
        s0=s0, r=r, sigma=sigma, maturity=maturity, steps=steps, n_paths=n_paths, seed=seed, antithetic=True
    )
    payoff_fn = lambda states, t: asian_payoff(states, t, strike)
    basis_fn = lambda states, t: asian_state_basis(states["spot"][:, t], states["average"][:, t], strike)
    mask_fn = lambda states, t: lockout_mask(states, t, lockout_time, maturity)

    in_result, policy = fit_tvr(
        states=fit_sim.states,
        payoff_fn=payoff_fn,
        basis_fn=basis_fn,
        exercise_mask_fn=mask_fn,
        r=r,
        maturity=maturity,
    )

    eval_sim = simulate_gbm_with_running_average(
        s0=s0, r=r, sigma=sigma, maturity=maturity, steps=steps, n_paths=n_paths, seed=seed + 7919, antithetic=True
    )
    out_price, out_stderr, stopping_times, _ = policy.evaluate(eval_sim.states, r)
    euro_like = float(np.exp(-r * maturity) * payoff_fn(eval_sim.states, steps).mean())

    summary = pd.DataFrame(
        [
            {
                "example": "TVR Asian shared extension",
                "spot0": s0,
                "strike": strike,
                "risk_free": r,
                "sigma": sigma,
                "maturity": maturity,
                "lockout_time": lockout_time,
                "exercise_dates": steps,
                "paths_total": n_paths,
                "tvr_price_in_sample": in_result.price,
                "tvr_price_out_of_sample": out_price,
                "tvr_stderr_out_of_sample": out_stderr,
                "european_style_mc": euro_like,
                "early_exercise_premium_vs_euro_mc": out_price - euro_like,
                "benchmark_note": "Compact shared extension study; no direct PDE benchmark included for the Asian case.",
            }
        ]
    )
    summary.to_csv(OUTPUT_DIR / "tvr_asian_results.csv", index=False)

    eval_t = int(steps * 0.5)
    intrinsic = payoff_fn(eval_sim.states, eval_t)
    exercised_at_t = stopping_times == eval_t
    plt.figure(figsize=(8, 6))
    plt.scatter(
        eval_sim.states["spot"][:4000, eval_t],
        eval_sim.states["average"][:4000, eval_t],
        c=intrinsic[:4000],
        cmap="plasma",
        s=10,
        alpha=0.6,
        label="State cloud",
    )
    if np.any(exercised_at_t):
        idx = np.where(exercised_at_t)[0][:1000]
        plt.scatter(
            eval_sim.states["spot"][idx, eval_t],
            eval_sim.states["average"][idx, eval_t],
            color="red",
            s=14,
            alpha=0.75,
            label="Exercised at t",
        )
    plt.colorbar(label="Intrinsic value at t")
    plt.xlabel("Spot")
    plt.ylabel("Running arithmetic average")
    plt.title("TVR Asian shared extension: state scatter and exercise overlay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "tvr_asian_state_scatter.png", dpi=180)
    plt.close()

    print(summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
