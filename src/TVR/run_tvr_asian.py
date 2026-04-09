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
from simulators import simulate_gbm_paths
from tvr_engine import fit_tvr


OUTPUT_DIR = PROJECT_ROOT / "outputs"
VISUALS_DIR = PROJECT_ROOT / "visuals"

STRIKE = 100.0
RISK_FREE = 0.06
SIGMA = 0.20
MATURITY = 2.0
LOCKOUT_TIME = 0.25
DATES_PER_YEAR = 100
STEPS = int(round(MATURITY * DATES_PER_YEAR))
PRE_AVERAGING_TIME = 0.25
TOTAL_PATHS = 50_000
BASE_SEED = 20260417
INITIAL_AVERAGES = [90.0, 100.0, 110.0]
SPOTS = [80.0, 90.0, 100.0, 110.0, 120.0]
REP_CASE = (100.0, 100.0)


def build_prestarted_average(spot: np.ndarray, initial_average: float, maturity: float, pre_time: float) -> np.ndarray:
    steps = spot.shape[1] - 1
    dt = maturity / steps
    avg = np.empty_like(spot)
    avg[:, 0] = initial_average
    cumulative_integral = np.cumsum(spot[:, 1:], axis=1) * dt
    for j in range(1, steps + 1):
        t = j * dt
        avg[:, j] = (pre_time * initial_average + cumulative_integral[:, j - 1]) / (pre_time + t)
    return avg


def asian_payoff(states: dict[str, np.ndarray], t: int, strike: float = STRIKE) -> np.ndarray:
    return np.maximum(strike - states["average"][:, t], 0.0)


def lockout_mask(states: dict[str, np.ndarray], t: int, lockout_time: float = LOCKOUT_TIME, maturity: float = MATURITY) -> np.ndarray:
    steps = states["spot"].shape[1] - 1
    time = maturity * t / steps
    return np.full(states["spot"].shape[0], time >= lockout_time, dtype=bool)


def prepare_states(spot0: float, initial_average: float, seed: int) -> dict[str, np.ndarray]:
    sim = simulate_gbm_paths(
        s0=spot0,
        r=RISK_FREE,
        sigma=SIGMA,
        maturity=MATURITY,
        steps=STEPS,
        n_paths=TOTAL_PATHS,
        seed=seed,
        antithetic=True,
    )
    average = build_prestarted_average(sim.states["spot"], initial_average, MATURITY, PRE_AVERAGING_TIME)
    return {"spot": sim.states["spot"], "average": average}


def representative_case_seed(initial_average: float, spot0: float) -> int:
    a_idx = INITIAL_AVERAGES.index(initial_average)
    s_idx = SPOTS.index(spot0)
    return BASE_SEED + 1000 * a_idx + 100 * s_idx


def run_benchmark() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    longstaff_path = OUTPUT_DIR / "longstaff_asian_benchmark.csv"
    longstaff_df = pd.read_csv(longstaff_path) if longstaff_path.exists() else None

    rows = []
    rep_states: dict[str, np.ndarray] | None = None
    rep_stopping: np.ndarray | None = None

    payoff_fn = lambda states, t: asian_payoff(states, t, strike=STRIKE)
    basis_fn = lambda states, t: asian_state_basis(states["spot"][:, t], states["average"][:, t], strike=STRIKE)

    for initial_average in INITIAL_AVERAGES:
        for spot0 in SPOTS:
            seed = representative_case_seed(initial_average, spot0)
            fit_states = prepare_states(spot0, initial_average, seed)
            in_result, policy = fit_tvr(
                states=fit_states,
                payoff_fn=payoff_fn,
                basis_fn=basis_fn,
                exercise_mask_fn=lockout_mask,
                r=RISK_FREE,
                maturity=MATURITY,
            )
            eval_seed = seed + 7919
            eval_states = prepare_states(spot0, initial_average, eval_seed)
            out_price, out_stderr, stopping_times, _ = policy.evaluate(eval_states, r=RISK_FREE)
            euro_mc = float(np.exp(-RISK_FREE * MATURITY) * asian_payoff(eval_states, STEPS).mean())
            longstaff_price = np.nan
            if longstaff_df is not None:
                match = longstaff_df[
                    (longstaff_df["initial_average"] == initial_average)
                    & (longstaff_df["spot0"] == spot0)
                ]
                if not match.empty:
                    longstaff_price = float(match.iloc[0]["american_asian_lsmc"])
            rows.append(
                {
                    "initial_average": initial_average,
                    "spot0": spot0,
                    "strike": STRIKE,
                    "risk_free": RISK_FREE,
                    "sigma": SIGMA,
                    "maturity": MATURITY,
                    "lockout_time": LOCKOUT_TIME,
                    "pre_averaging_time": PRE_AVERAGING_TIME,
                    "exercise_dates_per_year": DATES_PER_YEAR,
                    "exercise_steps": STEPS,
                    "paths_total": TOTAL_PATHS,
                    "american_asian_tvr_is": in_result.price,
                    "american_asian_tvr_oos": out_price,
                    "tvr_stderr": out_stderr,
                    "tvr_fit_minus_eval": in_result.price - out_price,
                    "european_asian_mc": euro_mc,
                    "tvr_early_exercise_premium": out_price - euro_mc,
                    "tvr_minus_lsmc": out_price - longstaff_price if np.isfinite(longstaff_price) else np.nan,
                    "fit_seed": seed,
                    "eval_seed": eval_seed,
                }
            )
            if (initial_average, spot0) == REP_CASE:
                rep_states = eval_states
                rep_stopping = stopping_times

    df = pd.DataFrame(rows).sort_values(["initial_average", "spot0"]).reset_index(drop=True)
    df.to_csv(OUTPUT_DIR / "tvr_asian_benchmark.csv", index=False)

    if rep_states is not None and rep_stopping is not None:
        eval_t = int(STEPS * 0.50)
        intrinsic = asian_payoff(rep_states, eval_t)
        exercised_at_t = rep_stopping == eval_t
        plt.figure(figsize=(8, 6))
        plt.scatter(
            rep_states["spot"][:4000, eval_t],
            rep_states["average"][:4000, eval_t],
            c=intrinsic[:4000],
            cmap="plasma",
            s=10,
            alpha=0.6,
            label="State cloud",
        )
        if np.any(exercised_at_t):
            idx = np.where(exercised_at_t)[0][:1000]
            plt.scatter(
                rep_states["spot"][idx, eval_t],
                rep_states["average"][idx, eval_t],
                color="red",
                s=14,
                alpha=0.75,
                label="Exercised at t",
            )
        plt.colorbar(label="Intrinsic value at t")
        plt.xlabel("Spot")
        plt.ylabel("Running arithmetic average")
        plt.title("Asian benchmark state scatter with exercise overlay: TVR")
        plt.legend()
        plt.tight_layout()
        plt.savefig(VISUALS_DIR / "tvr_asian_state_scatter.png", dpi=180)
        plt.close()

    print(df.round(6).to_string(index=False))
    return df


def main() -> None:
    run_benchmark()


if __name__ == "__main__":
    main()
