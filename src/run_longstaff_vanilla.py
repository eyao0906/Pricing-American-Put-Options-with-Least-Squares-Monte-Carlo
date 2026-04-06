from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from basis import vanilla_put_basis
from benchmarks import american_put_binomial_crr, american_put_implicit_fd, black_scholes_put
from lsm_engine import fit_lsm
from simulators import simulate_gbm_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
VISUALS_DIR = PROJECT_ROOT / "visuals"

STRIKE = 40.0
RISK_FREE = 0.06
SPOTS = [36.0, 38.0, 40.0, 42.0, 44.0]
SIGMAS = [0.20, 0.40]
MATURITIES = [1.0, 2.0]
DEFAULT_DATES_PER_YEAR = 50
DEFAULT_PATHS = 100_000
BASE_SEED = 20260406


def ensure_dirs() -> None:
    for directory in (DATA_DIR, OUTPUT_DIR, VISUALS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def put_payoff(states: dict[str, np.ndarray], t: int, strike: float = STRIKE) -> np.ndarray:
    return np.maximum(strike - states["spot"][:, t], 0.0)


def always_exercisable(states: dict[str, np.ndarray], t: int) -> np.ndarray:
    return np.ones(states["spot"].shape[0], dtype=bool)


def price_vanilla_lsm(
    spot: float,
    sigma: float,
    maturity: float,
    paths: int = DEFAULT_PATHS,
    dates_per_year: int = DEFAULT_DATES_PER_YEAR,
    basis_size: int = 4,
    seed: int = BASE_SEED,
):
    steps = int(round(maturity * dates_per_year))
    sim = simulate_gbm_paths(
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
    result, policy = fit_lsm(
        states=sim.states,
        payoff_fn=payoff_fn,
        basis_fn=basis_fn,
        exercise_mask_fn=always_exercisable,
        r=RISK_FREE,
        maturity=maturity,
    )
    return sim, result, policy


def benchmark_row(spot: float, sigma: float, maturity: float, seed: int = BASE_SEED) -> tuple[dict[str, float], list[dict[str, float]]]:
    _, lsm_result, _ = price_vanilla_lsm(spot=spot, sigma=sigma, maturity=maturity, seed=seed)
    european = black_scholes_put(spot, STRIKE, RISK_FREE, sigma, maturity)
    steps = int(round(maturity * DEFAULT_DATES_PER_YEAR))
    american_fd = american_put_implicit_fd(spot, STRIKE, RISK_FREE, sigma, maturity, exercise_steps=steps)
    american_crr = american_put_binomial_crr(spot, STRIKE, RISK_FREE, sigma, maturity, steps=max(500, steps * 10))

    row = {
        "spot": spot,
        "strike": STRIKE,
        "sigma": sigma,
        "maturity": maturity,
        "exercise_dates_per_year": DEFAULT_DATES_PER_YEAR,
        "paths_total": DEFAULT_PATHS,
        "european_put_bs": european,
        "american_put_fd": american_fd,
        "fd_early_exercise_premium": american_fd - european,
        "american_put_lsmc": lsm_result.price,
        "lsmc_stderr": lsm_result.stderr,
        "lsmc_early_exercise_premium": lsm_result.price - european,
        "lsmc_premium_minus_fd_premium": (lsm_result.price - european) - (american_fd - european),
        "american_put_crr": american_crr,
    }
    return row, lsm_result.exercise_summary


def run_benchmark_table() -> pd.DataFrame:
    rows = []
    boundary_rows: list[dict[str, float]] = []
    for spot in SPOTS:
        for sigma in SIGMAS:
            for maturity in MATURITIES:
                row, summaries = benchmark_row(spot, sigma, maturity)
                rows.append(row)
                for summary in summaries:
                    boundary_rows.append(
                        {
                            "spot0": spot,
                            "sigma": sigma,
                            "maturity": maturity,
                            **summary,
                        }
                    )
    df = pd.DataFrame(rows).sort_values(["sigma", "maturity", "spot"]).reset_index(drop=True)
    df.to_csv(OUTPUT_DIR / "longstaff_vanilla_benchmark.csv", index=False)
    pd.DataFrame(boundary_rows).to_csv(OUTPUT_DIR / "longstaff_vanilla_exercise_boundary_summary.csv", index=False)
    return df


def run_in_vs_out_sample() -> pd.DataFrame:
    cases = [(36.0, 0.20, 1.0), (40.0, 0.20, 2.0), (44.0, 0.40, 1.0), (40.0, 0.40, 2.0)]
    rows = []
    for case_id, (spot, sigma, maturity) in enumerate(cases):
        for repeat in range(3):
            seed_fit = BASE_SEED + 1000 * case_id + repeat
            seed_eval = BASE_SEED + 2000 * case_id + repeat
            _, in_result, policy = price_vanilla_lsm(spot=spot, sigma=sigma, maturity=maturity, seed=seed_fit)
            steps = int(round(maturity * DEFAULT_DATES_PER_YEAR))
            out_sim = simulate_gbm_paths(
                s0=spot,
                r=RISK_FREE,
                sigma=sigma,
                maturity=maturity,
                steps=steps,
                n_paths=DEFAULT_PATHS,
                seed=seed_eval,
                antithetic=True,
            )
            out_price, out_stderr, _, _ = policy.evaluate(out_sim.states, r=RISK_FREE)
            rows.append(
                {
                    "spot": spot,
                    "sigma": sigma,
                    "maturity": maturity,
                    "fit_seed": seed_fit,
                    "eval_seed": seed_eval,
                    "in_sample_price": in_result.price,
                    "in_sample_stderr": in_result.stderr,
                    "out_of_sample_price": out_price,
                    "out_of_sample_stderr": out_stderr,
                    "difference": in_result.price - out_price,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "longstaff_vanilla_in_vs_out_sample.csv", index=False)
    return df


def run_paths_sensitivity() -> pd.DataFrame:
    cases = [(40.0, 0.20, 1.0), (40.0, 0.40, 2.0)]
    path_counts = [25_000, 50_000, 100_000, 200_000]
    rows = []
    for spot, sigma, maturity in cases:
        benchmark = american_put_implicit_fd(
            spot, STRIKE, RISK_FREE, sigma, maturity, exercise_steps=int(round(maturity * DEFAULT_DATES_PER_YEAR))
        )
        for paths in path_counts:
            _, result, _ = price_vanilla_lsm(
                spot=spot,
                sigma=sigma,
                maturity=maturity,
                paths=paths,
                seed=BASE_SEED + paths,
            )
            rows.append(
                {
                    "spot": spot,
                    "sigma": sigma,
                    "maturity": maturity,
                    "paths_total": paths,
                    "lsmc_price": result.price,
                    "lsmc_stderr": result.stderr,
                    "fd_benchmark": benchmark,
                    "lsmc_minus_fd": result.price - benchmark,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "longstaff_vanilla_paths_sensitivity.csv", index=False)
    return df


def run_timestep_sensitivity() -> pd.DataFrame:
    cases = [(40.0, 0.20, 1.0), (40.0, 0.40, 2.0)]
    dates_per_year_list = [25, 50, 100]
    rows = []
    for spot, sigma, maturity in cases:
        for dates_per_year in dates_per_year_list:
            _, result, _ = price_vanilla_lsm(
                spot=spot,
                sigma=sigma,
                maturity=maturity,
                dates_per_year=dates_per_year,
                seed=BASE_SEED + dates_per_year,
            )
            benchmark = american_put_implicit_fd(
                spot, STRIKE, RISK_FREE, sigma, maturity, exercise_steps=int(round(maturity * dates_per_year))
            )
            rows.append(
                {
                    "spot": spot,
                    "sigma": sigma,
                    "maturity": maturity,
                    "exercise_dates_per_year": dates_per_year,
                    "lsmc_price": result.price,
                    "lsmc_stderr": result.stderr,
                    "fd_benchmark": benchmark,
                    "lsmc_minus_fd": result.price - benchmark,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "longstaff_vanilla_timestep_sensitivity.csv", index=False)
    return df


def run_basis_sensitivity() -> pd.DataFrame:
    cases = [(40.0, 0.20, 1.0), (40.0, 0.40, 2.0)]
    basis_sizes = [2, 3, 4, 5]
    rows = []
    for spot, sigma, maturity in cases:
        benchmark = american_put_implicit_fd(
            spot, STRIKE, RISK_FREE, sigma, maturity, exercise_steps=int(round(maturity * DEFAULT_DATES_PER_YEAR))
        )
        for basis_size in basis_sizes:
            _, result, _ = price_vanilla_lsm(
                spot=spot,
                sigma=sigma,
                maturity=maturity,
                basis_size=basis_size,
                seed=BASE_SEED + basis_size * 17,
            )
            rows.append(
                {
                    "spot": spot,
                    "sigma": sigma,
                    "maturity": maturity,
                    "basis_size": basis_size,
                    "lsmc_price": result.price,
                    "lsmc_stderr": result.stderr,
                    "fd_benchmark": benchmark,
                    "lsmc_minus_fd": result.price - benchmark,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "longstaff_vanilla_basis_sensitivity.csv", index=False)
    return df


def run_seed_sensitivity() -> pd.DataFrame:
    cases = [(36.0, 0.20, 1.0), (40.0, 0.20, 2.0), (44.0, 0.40, 1.0)]
    rows = []
    for spot, sigma, maturity in cases:
        benchmark = american_put_implicit_fd(
            spot, STRIKE, RISK_FREE, sigma, maturity, exercise_steps=int(round(maturity * DEFAULT_DATES_PER_YEAR))
        )
        case_prices = []
        seeds = [BASE_SEED + i for i in range(5)]
        for seed in seeds:
            _, result, _ = price_vanilla_lsm(spot=spot, sigma=sigma, maturity=maturity, seed=seed)
            case_prices.append(result.price)
            rows.append(
                {
                    "spot": spot,
                    "sigma": sigma,
                    "maturity": maturity,
                    "seed": seed,
                    "lsmc_price": result.price,
                    "lsmc_stderr": result.stderr,
                    "fd_benchmark": benchmark,
                    "bias_vs_fd": result.price - benchmark,
                }
            )
        rows.append(
            {
                "spot": spot,
                "sigma": sigma,
                "maturity": maturity,
                "seed": "summary",
                "lsmc_price": float(np.mean(case_prices)),
                "lsmc_stderr": float(np.std(case_prices, ddof=1)),
                "fd_benchmark": benchmark,
                "bias_vs_fd": float(np.mean(np.array(case_prices) - benchmark)),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "longstaff_vanilla_seed_sensitivity.csv", index=False)
    return df


def save_config_snapshot() -> None:
    records = []
    for spot in SPOTS:
        for sigma in SIGMAS:
            for maturity in MATURITIES:
                records.append(
                    {
                        "spot": spot,
                        "strike": STRIKE,
                        "sigma": sigma,
                        "maturity": maturity,
                        "risk_free": RISK_FREE,
                        "dates_per_year": DEFAULT_DATES_PER_YEAR,
                        "paths_total": DEFAULT_PATHS,
                        "paths_base": DEFAULT_PATHS // 2,
                        "paths_antithetic": DEFAULT_PATHS // 2,
                        "basis_size": 4,
                    }
                )
    pd.DataFrame(records).to_csv(DATA_DIR / "longstaff_vanilla_config.csv", index=False)


def plot_benchmark(df: pd.DataFrame) -> None:
    labels = [f"S={s:.0f},σ={sig:.2f},T={t:.0f}" for s, sig, t in zip(df["spot"], df["sigma"], df["maturity"])]
    x = np.arange(len(df))
    plt.figure(figsize=(12, 6))
    plt.plot(x, df["american_put_fd"], marker="o", label="FD benchmark")
    plt.plot(x, df["american_put_lsmc"], marker="s", label="LSMC")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("American put price")
    plt.title("Vanilla American put: LSMC vs finite-difference benchmark")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "longstaff_vanilla_benchmark_comparison.png", dpi=180)
    plt.close()


def plot_boundary() -> None:
    boundary_df = pd.read_csv(OUTPUT_DIR / "longstaff_vanilla_exercise_boundary_summary.csv")
    reps = boundary_df[(boundary_df["spot0"] == 40.0) & (boundary_df["maturity"].isin([1.0, 2.0]))]
    plt.figure(figsize=(9, 6))
    for sigma in sorted(reps["sigma"].unique()):
        for maturity in sorted(reps["maturity"].unique()):
            subset = reps[(reps["sigma"] == sigma) & (reps["maturity"] == maturity)]
            plt.plot(subset["time"], subset["boundary_mean"], marker="o", label=f"σ={sigma:.2f}, T={maturity:.0f}")
    plt.xlabel("Time")
    plt.ylabel("Mean exercise spot among exercised paths")
    plt.title("Representative empirical exercise boundary summaries")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "longstaff_vanilla_exercise_boundary.png", dpi=180)
    plt.close()


def plot_robustness(paths_df: pd.DataFrame, basis_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    for key, subset in paths_df.groupby(["spot", "sigma", "maturity"]):
        plt.plot(subset["paths_total"], subset["lsmc_minus_fd"], marker="o", label=f"S={key[0]:.0f},σ={key[1]:.2f},T={key[2]:.0f}")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Total paths")
    plt.ylabel("LSMC - FD")
    plt.title("Path-count sensitivity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "longstaff_vanilla_paths_sensitivity.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for key, subset in basis_df.groupby(["spot", "sigma", "maturity"]):
        plt.plot(subset["basis_size"], subset["lsmc_minus_fd"], marker="o", label=f"S={key[0]:.0f},σ={key[1]:.2f},T={key[2]:.0f}")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Basis size")
    plt.ylabel("LSMC - FD")
    plt.title("Basis-size sensitivity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "longstaff_vanilla_basis_sensitivity.png", dpi=180)
    plt.close()


def main() -> None:
    ensure_dirs()
    save_config_snapshot()
    benchmark_df = run_benchmark_table()
    inout_df = run_in_vs_out_sample()
    paths_df = run_paths_sensitivity()
    run_timestep_sensitivity()
    basis_df = run_basis_sensitivity()
    run_seed_sensitivity()
    plot_benchmark(benchmark_df)
    plot_boundary()
    plot_robustness(paths_df, basis_df)

    print("Created vanilla benchmark and robustness outputs:")
    print(benchmark_df.round(6).to_string(index=False))
    print(f"In/out-sample rows: {len(inout_df)}")


if __name__ == "__main__":
    main()
