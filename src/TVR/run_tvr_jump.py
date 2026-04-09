from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def infer_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "src").exists() or (candidate / "outputs").exists() or (candidate / "visuals").exists():
            return candidate
    return here.parent


PROJECT_ROOT = infer_project_root()
LONGSTAFF_DIR = PROJECT_ROOT / "src" / "LongstaffMethod"
if LONGSTAFF_DIR.exists() and str(LONGSTAFF_DIR) not in sys.path:
    sys.path.append(str(LONGSTAFF_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent))

from basis import vanilla_put_basis
from simulators import SimulationResult, simulate_gbm_paths
from tvr_engine import fit_tvr

OUTPUT_DIR = PROJECT_ROOT / "outputs"
VISUALS_DIR = PROJECT_ROOT / "visuals"

STRIKE = 40.0
RISK_FREE = 0.06
MATURITY = 1.0
DATES_PER_YEAR = 26
STEPS = int(round(MATURITY * DATES_PER_YEAR))
TOTAL_PATHS = 50_000
BASE_SEED = 20260430
SPOTS = [36.0, 38.0, 40.0, 42.0, 44.0]
REP_SPOT = 40.0
MODELS = [
    {
        "model": "no_jump_matched_var",
        "sigma": 0.30,
        "jump_intensity": 0.00,
        "label": r"No jump ($\sigma=0.30,\ \lambda=0.00$)",
    },
    {
        "model": "jump_to_ruin_matched_var",
        "sigma": 0.20,
        "jump_intensity": 0.05,
        "label": r"Jump-to-ruin ($\sigma=0.20,\ \lambda=0.05$)",
    },
]


def put_payoff(states: dict[str, np.ndarray], t: int, strike: float = STRIKE) -> np.ndarray:
    return np.maximum(strike - states["spot"][:, t], 0.0)


def always_exercisable(states: dict[str, np.ndarray], t: int) -> np.ndarray:
    return np.ones(states["spot"].shape[0], dtype=bool)


def simulate_risk_neutral_jump_to_ruin_paths(
    s0: float,
    r: float,
    sigma: float,
    maturity: float,
    steps: int,
    n_paths: int,
    jump_intensity: float,
    seed: int,
    antithetic: bool = True,
) -> SimulationResult:
    if antithetic and n_paths % 2 != 0:
        raise ValueError("n_paths must be even when antithetic=True")
    base_paths = n_paths // 2 if antithetic else n_paths
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((base_paths, steps))
    z_full = np.vstack([z, -z]) if antithetic else z
    jump_uniforms = rng.random((n_paths, steps))
    jump_occurs = jump_uniforms < 1.0 - np.exp(-jump_intensity * maturity / steps)

    dt = maturity / steps
    times = np.linspace(0.0, maturity, steps + 1)
    drift = (r + jump_intensity - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    spot = np.empty((n_paths, steps + 1), dtype=float)
    spot[:, 0] = s0
    ruined = np.zeros(n_paths, dtype=bool)
    for t in range(steps):
        alive = ~ruined
        next_values = np.zeros(n_paths, dtype=float)
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
            "n_paths": n_paths,
            "jump_intensity": jump_intensity,
            "seed": seed,
            "antithetic": int(antithetic),
        },
    )


def simulate_case(spot0: float, sigma: float, jump_intensity: float, seed: int) -> SimulationResult:
    if jump_intensity == 0.0:
        return simulate_gbm_paths(
            s0=spot0,
            r=RISK_FREE,
            sigma=sigma,
            maturity=MATURITY,
            steps=STEPS,
            n_paths=TOTAL_PATHS,
            seed=seed,
            antithetic=True,
        )
    return simulate_risk_neutral_jump_to_ruin_paths(
        s0=spot0,
        r=RISK_FREE,
        sigma=sigma,
        maturity=MATURITY,
        steps=STEPS,
        n_paths=TOTAL_PATHS,
        jump_intensity=jump_intensity,
        seed=seed,
        antithetic=True,
    )


def fit_and_eval_tvr(spot0: float, sigma: float, jump_intensity: float, seed: int):
    payoff_fn = lambda states, t: put_payoff(states, t, strike=STRIKE)
    basis_fn = lambda states, t: vanilla_put_basis(states["spot"][:, t], strike=STRIKE, basis_size=4)

    fit_states = simulate_case(spot0, sigma, jump_intensity, seed)
    in_result, policy = fit_tvr(
        states=fit_states.states,
        payoff_fn=payoff_fn,
        basis_fn=basis_fn,
        exercise_mask_fn=always_exercisable,
        r=RISK_FREE,
        maturity=MATURITY,
    )
    eval_seed = seed + 7919
    eval_states = simulate_case(spot0, sigma, jump_intensity, eval_seed)
    out_price, out_stderr, _, _ = policy.evaluate(eval_states.states, r=RISK_FREE)
    euro_mc = float(np.exp(-RISK_FREE * MATURITY) * put_payoff(eval_states.states, STEPS).mean())
    diagnostics = {
        "fit_seed": seed,
        "eval_seed": eval_seed,
        "fit_price": in_result.price,
        "fit_stderr": in_result.stderr,
        "eval_price": out_price,
        "eval_stderr": out_stderr,
        "european_mc": euro_mc,
    }
    return in_result, diagnostics, policy


def case_seed(model_idx: int, spot_idx: int) -> int:
    return BASE_SEED + 1000 * model_idx + 100 * spot_idx + 1


def regression_boundary_series(policy, strike: float = STRIKE, upper_mult: float = 1.25, grid_size: int = 1000) -> pd.DataFrame:
    grid = np.linspace(1e-6 * strike, upper_mult * strike, grid_size)
    rows: list[dict[str, float]] = []
    dt = policy.dt
    for t, beta in sorted(policy.regression_map.items()):
        beta = np.asarray(beta, dtype=float)
        basis = vanilla_put_basis(grid, strike=strike, basis_size=beta.shape[0])
        continuation = basis @ beta
        intrinsic = np.maximum(strike - grid, 0.0)
        itm = grid <= strike
        diff = continuation[itm] - intrinsic[itm]
        g = grid[itm]
        idx = np.where(diff <= 0.0)[0]
        if idx.size == 0:
            boundary = np.nan
        else:
            k = idx[-1]
            boundary = float(g[k])
            if k < len(g) - 1 and diff[k] <= 0.0 <= diff[k + 1] and diff[k + 1] != diff[k]:
                x1, x2 = g[k], g[k + 1]
                y1, y2 = diff[k], diff[k + 1]
                boundary = float(x1 + (0.0 - y1) * (x2 - x1) / (y2 - y1))
        rows.append(
            {
                "time_index": float(t),
                "time": float(t * dt),
                "boundary": boundary,
                "boundary_ratio": boundary / strike if np.isfinite(boundary) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_benchmark() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    longstaff_path = OUTPUT_DIR / "longstaff_jump_benchmark.csv"
    longstaff_df = pd.read_csv(longstaff_path) if longstaff_path.exists() else None

    rows = []
    rep_boundaries: dict[str, pd.DataFrame] = {}

    for model_idx, spec in enumerate(MODELS):
        for spot_idx, spot0 in enumerate(SPOTS):
            seed = case_seed(model_idx, spot_idx)
            in_result, diagnostics, policy = fit_and_eval_tvr(spot0, spec["sigma"], spec["jump_intensity"], seed)
            longstaff_price = np.nan
            if longstaff_df is not None:
                match = longstaff_df[(longstaff_df["model"] == spec["model"]) & (longstaff_df["spot0"] == spot0)]
                if not match.empty:
                    longstaff_price = float(match.iloc[0]["american_jump_lsmc"])
            rows.append(
                {
                    "model": spec["model"],
                    "spot0": spot0,
                    "strike": STRIKE,
                    "risk_free": RISK_FREE,
                    "sigma": spec["sigma"],
                    "maturity": MATURITY,
                    "jump_intensity": spec["jump_intensity"],
                    "exercise_dates_per_year": DATES_PER_YEAR,
                    "exercise_steps": STEPS,
                    "paths_total": TOTAL_PATHS,
                    "american_jump_tvr_is": diagnostics["fit_price"],
                    "american_jump_tvr_oos": diagnostics["eval_price"],
                    "tvr_stderr": diagnostics["eval_stderr"],
                    "tvr_fit_minus_eval": diagnostics["fit_price"] - diagnostics["eval_price"],
                    "european_jump_mc": diagnostics["european_mc"],
                    "tvr_early_exercise_premium": diagnostics["eval_price"] - diagnostics["european_mc"],
                    "tvr_minus_lsmc": diagnostics["eval_price"] - longstaff_price if np.isfinite(longstaff_price) else np.nan,
                    "fit_seed": diagnostics["fit_seed"],
                    "eval_seed": diagnostics["eval_seed"],
                }
            )
            if spot0 == REP_SPOT:
                rep_boundaries[spec["model"]] = regression_boundary_series(policy)

    df = pd.DataFrame(rows).sort_values(["model", "spot0"]).reset_index(drop=True)
    df.to_csv(OUTPUT_DIR / "tvr_jump_benchmark.csv", index=False)
    df.to_csv(OUTPUT_DIR / "tvr_jump_results.csv", index=False)

    plt.figure(figsize=(8, 5))
    for spec in MODELS:
        boundary_df = rep_boundaries.get(spec["model"])
        if boundary_df is not None and not boundary_df.empty:
            plt.plot(boundary_df["time"], boundary_df["boundary_ratio"], marker="o", label=spec["label"])
            boundary_df.assign(model=spec["model"]).to_csv(OUTPUT_DIR / f"{spec['model']}_tvr_jump_boundary.csv", index=False)
    plt.xlabel("Time")
    plt.ylabel(r"Critical exercise boundary $S^*/K$")
    plt.title("Paper-inspired matched jump benchmark: TVR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "tvr_jump_boundary_comparison.png", dpi=180)
    plt.close()

    print(df.round(6).to_string(index=False))
    return df


def main() -> None:
    run_benchmark()


if __name__ == "__main__":
    main()
