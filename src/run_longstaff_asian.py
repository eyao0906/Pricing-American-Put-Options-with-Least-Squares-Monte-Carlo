from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from basis import asian_state_basis
from lsm_engine import fit_lsm
from simulators import simulate_gbm_with_running_average


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    seed = 20260416

    sim = simulate_gbm_with_running_average(
        s0=s0,
        r=r,
        sigma=sigma,
        maturity=maturity,
        steps=steps,
        n_paths=n_paths,
        seed=seed,
        antithetic=True,
    )

    payoff_fn = lambda states, t: asian_payoff(states, t, strike=strike)
    basis_fn = lambda states, t: asian_state_basis(states["spot"][:, t], states["average"][:, t], strike=strike)
    exercise_mask_fn = lambda states, t: lockout_mask(states, t, lockout_time=lockout_time, maturity=maturity)

    result, _ = fit_lsm(
        states=sim.states,
        payoff_fn=payoff_fn,
        basis_fn=basis_fn,
        exercise_mask_fn=exercise_mask_fn,
        r=r,
        maturity=maturity,
    )

    euro_like = float(np.exp(-r * maturity) * payoff_fn(sim.states, steps).mean())
    summary = pd.DataFrame(
        [
            {
                "example": "American-Bermuda-Asian",
                "spot0": s0,
                "strike": strike,
                "risk_free": r,
                "sigma": sigma,
                "maturity": maturity,
                "lockout_time": lockout_time,
                "exercise_dates": steps,
                "paths_total": n_paths,
                "lsm_price": result.price,
                "lsm_stderr": result.stderr,
                "european_style_mc": euro_like,
                "early_exercise_premium_vs_euro_mc": result.price - euro_like,
                "benchmark_note": "No direct PDE benchmark included here; benchmark emphasis is on the vanilla put case.",
            }
        ]
    )
    summary.to_csv(OUTPUT_DIR / "longstaff_asian_results.csv", index=False)

    sample_paths = 20
    plt.figure(figsize=(10, 6))
    t = sim.times
    for i in range(sample_paths):
        plt.plot(t, sim.states["spot"][i], alpha=0.6, linewidth=1)
        plt.plot(t, sim.states["average"][i], alpha=0.35, linewidth=1, linestyle="--")
    plt.axvline(lockout_time, color="black", linestyle=":", label="Lockout end")
    plt.title("Example 2: spot paths (solid) and running averages (dashed)")
    plt.xlabel("Time")
    plt.ylabel("Level")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "longstaff_asian_paths_example.png", dpi=180)
    plt.close()

    eval_t = int(steps * 0.5)
    intrinsic = payoff_fn(sim.states, eval_t)
    exercised_at_t = result.stopping_times == eval_t
    plt.figure(figsize=(8, 6))
    plt.scatter(
        sim.states["spot"][:4000, eval_t],
        sim.states["average"][:4000, eval_t],
        c=intrinsic[:4000],
        cmap="viridis",
        s=10,
        alpha=0.6,
        label="State cloud",
    )
    if np.any(exercised_at_t):
        idx = np.where(exercised_at_t)[0][:1000]
        plt.scatter(
            sim.states["spot"][idx, eval_t],
            sim.states["average"][idx, eval_t],
            color="red",
            s=14,
            alpha=0.75,
            label="Exercised at t",
        )
    plt.colorbar(label="Intrinsic value at t")
    plt.xlabel("Spot")
    plt.ylabel("Running arithmetic average")
    plt.title("Example 2 state scatter with exercise overlay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "longstaff_asian_state_scatter.png", dpi=180)
    plt.close()

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
