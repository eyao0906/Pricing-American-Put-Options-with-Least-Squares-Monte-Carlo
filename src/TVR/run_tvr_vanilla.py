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
from benchmarks import american_put_binomial_crr, american_put_implicit_fd, black_scholes_put
from simulators import simulate_gbm_paths
from tvr_engine import fit_tvr


OUTPUT_DIR = PROJECT_ROOT / "outputs"
VISUALS_DIR = PROJECT_ROOT / "visuals"

STRIKE = 40.0
RISK_FREE = 0.06
SPOTS = [36.0, 38.0, 40.0, 42.0, 44.0]
SIGMAS = [0.20, 0.40]
MATURITIES = [1.0, 2.0]
DEFAULT_DATES_PER_YEAR = 50
DEFAULT_PATHS = 100_000
BASE_SEED = 20260407


def put_payoff(states: dict[str, np.ndarray], t: int, strike: float = STRIKE) -> np.ndarray:
    return np.maximum(strike - states["spot"][:, t], 0.0)


def always_exercisable(states: dict[str, np.ndarray], t: int) -> np.ndarray:
    return np.ones(states["spot"].shape[0], dtype=bool)


def price_tvr_vanilla(
    spot: float,
    sigma: float,
    maturity: float,
    paths: int = DEFAULT_PATHS,
    dates_per_year: int = DEFAULT_DATES_PER_YEAR,
    basis_size: int = 4,
    seed: int = BASE_SEED,
):
    steps = int(round(maturity * dates_per_year))
    fit_sim = simulate_gbm_paths(
        s0=spot,
        r=RISK_FREE,
        sigma=sigma,
        maturity=maturity,
        steps=steps,
        n_paths=paths,
        seed=seed,
        antithetic=True,
    )
    payoff_fn = lambda states, t: put_payoff(states, t, strike=STRIKE)
    basis_fn = lambda states, t: vanilla_put_basis(states["spot"][:, t], strike=STRIKE, basis_size=basis_size)

    in_result, policy = fit_tvr(
        states=fit_sim.states,
        payoff_fn=payoff_fn,
        basis_fn=basis_fn,
        exercise_mask_fn=always_exercisable,
        r=RISK_FREE,
        maturity=maturity,
    )

    eval_seed = seed + 7919
    eval_sim = simulate_gbm_paths(
        s0=spot,
        r=RISK_FREE,
        sigma=sigma,
        maturity=maturity,
        steps=steps,
        n_paths=paths,
        seed=eval_seed,
        antithetic=True,
    )
    out_price, out_stderr, stopping_times, discounted = policy.evaluate(eval_sim.states, r=RISK_FREE)
    diagnostics = {
        "fit_seed": seed,
        "eval_seed": eval_seed,
        "fit_price": in_result.price,
        "fit_stderr": in_result.stderr,
        "out_price": out_price,
        "out_stderr": out_stderr,
    }
    return fit_sim, eval_sim, in_result, policy, stopping_times, discounted, diagnostics


def run_benchmark() -> pd.DataFrame:
    longstaff = pd.read_csv(OUTPUT_DIR / "longstaff_vanilla_benchmark.csv")
    rows = []
    for spot in SPOTS:
        for sigma in SIGMAS:
            for maturity in MATURITIES:
                _, _, in_result, _, _, _, diag = price_tvr_vanilla(spot, sigma, maturity)
                euro = black_scholes_put(spot, STRIKE, RISK_FREE, sigma, maturity)
                steps = int(round(maturity * DEFAULT_DATES_PER_YEAR))
                fd = american_put_implicit_fd(spot, STRIKE, RISK_FREE, sigma, maturity, exercise_steps=steps)
                crr = american_put_binomial_crr(spot, STRIKE, RISK_FREE, sigma, maturity, steps=max(500, steps * 10))
                lsm_row = longstaff[(longstaff["spot"] == spot) & (longstaff["sigma"] == sigma) & (longstaff["maturity"] == maturity)].iloc[0]
                rows.append(
                    {
                        "spot": spot,
                        "strike": STRIKE,
                        "sigma": sigma,
                        "maturity": maturity,
                        "european_put_bs": euro,
                        "american_put_fd": fd,
                        "american_put_crr": crr,
                        "tvr_price_in_sample": diag["fit_price"],
                        "tvr_price_out_of_sample": diag["out_price"],
                        "tvr_stderr_out_of_sample": diag["out_stderr"],
                        "tvr_minus_fd": diag["out_price"] - fd,
                        "tvr_minus_lsm": diag["out_price"] - float(lsm_row["american_put_lsmc"]),
                        "fit_seed": diag["fit_seed"],
                        "eval_seed": diag["eval_seed"],
                    }
                )
    df = pd.DataFrame(rows).sort_values(["sigma", "maturity", "spot"]).reset_index(drop=True)
    df.to_csv(OUTPUT_DIR / "tvr_vanilla_benchmark.csv", index=False)
    return df


def run_in_vs_out() -> pd.DataFrame:
    cases = [(36.0, 0.20, 1.0), (40.0, 0.20, 2.0), (44.0, 0.40, 1.0), (40.0, 0.40, 2.0)]
    rows = []
    for case_id, (spot, sigma, maturity) in enumerate(cases):
        for repeat in range(3):
            fit_seed = BASE_SEED + 1000 * case_id + repeat
            _, _, in_result, policy, _, _, _ = price_tvr_vanilla(spot, sigma, maturity, seed=fit_seed)
            steps = int(round(maturity * DEFAULT_DATES_PER_YEAR))
            eval_seed = BASE_SEED + 5000 + 2000 * case_id + repeat
            eval_sim = simulate_gbm_paths(
                s0=spot,
                r=RISK_FREE,
                sigma=sigma,
                maturity=maturity,
                steps=steps,
                n_paths=DEFAULT_PATHS,
                seed=eval_seed,
                antithetic=True,
            )
            out_price, out_stderr, _, _ = policy.evaluate(eval_sim.states, r=RISK_FREE)
            rows.append(
                {
                    "spot": spot,
                    "sigma": sigma,
                    "maturity": maturity,
                    "fit_seed": fit_seed,
                    "eval_seed": eval_seed,
                    "in_sample_price": in_result.price,
                    "out_of_sample_price": out_price,
                    "difference": in_result.price - out_price,
                    "stderr_out_of_sample": out_stderr,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "tvr_vanilla_in_vs_out_sample.csv", index=False)
    return df


def run_paths_sensitivity() -> pd.DataFrame:
    cases = [(40.0, 0.20, 1.0), (40.0, 0.40, 2.0)]
    path_counts = [25_000, 50_000, 100_000, 200_000]
    rows = []
    for spot, sigma, maturity in cases:
        fd = american_put_implicit_fd(spot, STRIKE, RISK_FREE, sigma, maturity, exercise_steps=int(round(maturity * DEFAULT_DATES_PER_YEAR)))
        for paths in path_counts:
            _, _, _, _, _, _, diag = price_tvr_vanilla(spot, sigma, maturity, paths=paths, seed=BASE_SEED + paths)
            rows.append(
                {
                    "spot": spot,
                    "sigma": sigma,
                    "maturity": maturity,
                    "paths_total": paths,
                    "tvr_price_out_of_sample": diag["out_price"],
                    "tvr_stderr_out_of_sample": diag["out_stderr"],
                    "fd_benchmark": fd,
                    "tvr_minus_fd": diag["out_price"] - fd,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "tvr_vanilla_paths_sensitivity.csv", index=False)
    return df


def run_timestep_sensitivity() -> pd.DataFrame:
    cases = [(40.0, 0.20, 1.0), (40.0, 0.40, 2.0)]
    rows = []
    for spot, sigma, maturity in cases:
        for dates_per_year in [25, 50, 100]:
            _, _, _, _, _, _, diag = price_tvr_vanilla(
                spot, sigma, maturity, dates_per_year=dates_per_year, seed=BASE_SEED + dates_per_year
            )
            fd = american_put_implicit_fd(spot, STRIKE, RISK_FREE, sigma, maturity, exercise_steps=int(round(maturity * dates_per_year)))
            rows.append(
                {
                    "spot": spot,
                    "sigma": sigma,
                    "maturity": maturity,
                    "exercise_dates_per_year": dates_per_year,
                    "tvr_price_out_of_sample": diag["out_price"],
                    "tvr_stderr_out_of_sample": diag["out_stderr"],
                    "fd_benchmark": fd,
                    "tvr_minus_fd": diag["out_price"] - fd,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "tvr_vanilla_timestep_sensitivity.csv", index=False)
    return df


def run_basis_sensitivity() -> pd.DataFrame:
    cases = [(40.0, 0.20, 1.0), (40.0, 0.40, 2.0)]
    rows = []
    for spot, sigma, maturity in cases:
        fd = american_put_implicit_fd(spot, STRIKE, RISK_FREE, sigma, maturity, exercise_steps=int(round(maturity * DEFAULT_DATES_PER_YEAR)))
        for basis_size in [2, 3, 4, 5]:
            _, _, _, _, _, _, diag = price_tvr_vanilla(
                spot, sigma, maturity, basis_size=basis_size, seed=BASE_SEED + basis_size * 17
            )
            rows.append(
                {
                    "spot": spot,
                    "sigma": sigma,
                    "maturity": maturity,
                    "basis_size": basis_size,
                    "tvr_price_out_of_sample": diag["out_price"],
                    "tvr_stderr_out_of_sample": diag["out_stderr"],
                    "fd_benchmark": fd,
                    "tvr_minus_fd": diag["out_price"] - fd,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "tvr_vanilla_basis_sensitivity.csv", index=False)
    return df


def plot_visuals(benchmark_df: pd.DataFrame) -> None:
    x = np.arange(len(benchmark_df))
    labels = [f"S={s:.0f},σ={sig:.2f},T={t:.0f}" for s, sig, t in zip(benchmark_df["spot"], benchmark_df["sigma"], benchmark_df["maturity"])]
    plt.figure(figsize=(12, 6))
    plt.plot(x, benchmark_df["american_put_fd"], marker="o", label="FD benchmark")
    plt.plot(x, benchmark_df["tvr_price_out_of_sample"], marker="s", label="TVR out-of-sample")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("American put price")
    plt.title("TVR vanilla benchmark: out-of-sample price vs FD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "tvr_vanilla_benchmark_comparison.png", dpi=180)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_df = run_benchmark()
    run_in_vs_out()
    run_paths_sensitivity()
    run_timestep_sensitivity()
    run_basis_sensitivity()
    plot_visuals(benchmark_df)
    print(benchmark_df.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
