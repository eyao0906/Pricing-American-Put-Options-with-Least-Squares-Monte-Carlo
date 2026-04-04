from pathlib import Path
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from data_loader import fetch_market_data, estimate_inputs
except Exception:  # pragma: no cover - allows cached/local runs without yfinance
    fetch_market_data = None
    estimate_inputs = None

from lsmc import price_american_put_lsmc
from tsitsiklis_vanroy import price_american_put_tvr


DEFAULT_TICKER = "AAPL"
DEFAULT_PERIOD = "5y"
DEFAULT_STEPS = 50
DEFAULT_N_PATHS = 100_000
DEFAULT_MATURITIES = [1.0]
DEFAULT_STRIKE_MULTIPLIERS = [0.9, 1.0, 1.1]


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_european_put(s0: float, k: float, r: float, sigma: float, t: float) -> float:
    if t <= 0:
        return max(k - s0, 0.0)
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return k * math.exp(-r * t) * norm_cdf(-d2) - s0 * norm_cdf(-d1)


def price_american_put_binomial(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    steps: int,
) -> float:
    """Cox-Ross-Rubinstein binomial tree for an American put."""
    if t <= 0:
        return max(k - s0, 0.0)
    if steps < 1:
        raise ValueError("steps must be at least 1")

    dt = t / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp(r * dt) - d) / (u - d)

    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"Risk-neutral probability out of bounds: p={p:.6f}. "
            "Increase the number of tree steps or review inputs."
        )

    j = np.arange(steps + 1)
    stock = s0 * (u ** j) * (d ** (steps - j))
    option = np.maximum(k - stock, 0.0)

    for n in range(steps - 1, -1, -1):
        option = disc * (p * option[1:] + (1.0 - p) * option[:-1])
        j = np.arange(n + 1)
        stock = s0 * (u ** j) * (d ** (n - j))
        intrinsic = np.maximum(k - stock, 0.0)
        option = np.maximum(option, intrinsic)

    return float(option[0])


def _read_cached_prices(ticker: str, data_dir: str) -> pd.DataFrame:
    candidates = [
        Path(data_dir) / f"{ticker.lower()}_prices.csv",
        Path(f"{ticker.lower()}_prices.csv"),
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            date_col = next((c for c in df.columns if c.lower() == "date"), None)
            if date_col is not None:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
            if "close" not in df.columns:
                close_col = next((c for c in df.columns if c.lower() == "close"), None)
                if close_col is None:
                    raise ValueError(f"Cached price file {path} does not contain a close column.")
                df = df.rename(columns={close_col: "close"})
            return df[["close"]].dropna()
    raise FileNotFoundError(f"No cached price file found for {ticker}.")


def _fallback_risk_free_rate(default: float = 0.04) -> float:
    candidates = [Path("data/pricing_results.csv"), Path("pricing_results.csv")]
    for path in candidates:
        if path.exists():
            try:
                prior = pd.read_csv(path)
                if "risk_free" in prior.columns and len(prior) > 0:
                    return float(prior["risk_free"].iloc[0])
            except Exception:
                pass
    return default


def load_market_inputs(ticker: str, period: str, data_dir: str) -> tuple[pd.DataFrame, dict]:
    if fetch_market_data is not None:
        try:
            prices = fetch_market_data(ticker=ticker, period=period, data_dir=data_dir)
        except Exception:
            prices = _read_cached_prices(ticker=ticker, data_dir=data_dir)
    else:
        prices = _read_cached_prices(ticker=ticker, data_dir=data_dir)

    try:
        if estimate_inputs is None:
            raise RuntimeError("estimate_inputs unavailable")
        inputs = estimate_inputs(prices)
    except Exception:
        log_ret = np.log(prices["close"] / prices["close"].shift(1)).dropna()
        inputs = {
            "s0": float(prices["close"].iloc[-1]),
            "sigma": float(log_ret.std()) * np.sqrt(252),
            "r": _fallback_risk_free_rate(),
        }

    return prices, inputs


def plot_historical_price(prices: pd.DataFrame, ticker: str, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(prices.index, prices["close"], label=f"{ticker} close")
    plt.title(f"{ticker} historical close (5y)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "historical_price.png", dpi=180)
    plt.close()


def plot_exercise_boundaries(boundaries: dict, steps: int, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    x = np.arange(1, steps + 1)
    for (maturity, strike), boundary in boundaries.items():
        label = f"K={strike:.2f}, T={maturity:.2f}"
        plt.plot(x, boundary[1:], label=label)
    plt.title("Estimated LSMC exercise boundary (American put)")
    plt.xlabel("Time step")
    plt.ylabel("Boundary stock price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "exercise_boundary.png", dpi=180)
    plt.close()


def plot_model_comparison(out_df: pd.DataFrame, output_dir: Path) -> None:
    methods = [
        ("european_put_bs", "European BS"),
        ("american_put_binomial", "Binomial American"),
        ("lsmc_price", "LSMC"),
        ("tvr_price", "TVR"),
    ]

    labels = []
    for _, row in out_df.iterrows():
        labels.append(f"T={row['maturity_years']:.2f}\nK={row['strike']:.1f}")

    idx = np.arange(len(out_df))
    width = 0.2
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(methods))

    plt.figure(figsize=(max(9, 1.8 * len(out_df)), 5.5))
    for offset, (col, name) in zip(offsets, methods):
        plt.bar(idx + offset, out_df[col], width, label=name)

    plt.xticks(idx, labels)
    plt.ylabel("Option price")
    plt.title("European BS vs Binomial American vs LSMC vs TVR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close()


def main() -> None:
    ticker = DEFAULT_TICKER
    maturities = DEFAULT_MATURITIES
    steps = DEFAULT_STEPS
    n_paths = DEFAULT_N_PATHS
    strike_multipliers = DEFAULT_STRIKE_MULTIPLIERS

    data_dir = Path("data")
    visuals_dir = Path("visuals")
    data_dir.mkdir(exist_ok=True)
    visuals_dir.mkdir(exist_ok=True)

    prices, inputs = load_market_inputs(ticker=ticker, period=DEFAULT_PERIOD, data_dir=str(data_dir))
    s0, sigma, r = inputs["s0"], inputs["sigma"], inputs["r"]

    strikes = [multiplier * s0 for multiplier in strike_multipliers]
    records = []
    boundaries = {}

    for maturity in maturities:
        for k in strikes:
            euro_put = bs_european_put(s0, k, r, sigma, maturity)
            binomial_put = price_american_put_binomial(s0, k, r, sigma, maturity, steps)

            lsmc = price_american_put_lsmc(
                s0=s0,
                k=k,
                r=r,
                sigma=sigma,
                t=maturity,
                steps=steps,
                n_paths=n_paths,
                basis_degree=3,
                seed=42,
            )
            tvr = price_american_put_tvr(
                s0=s0,
                k=k,
                r=r,
                sigma=sigma,
                t=maturity,
                steps=steps,
                n_paths=n_paths,
                seed=43,
            )

            boundaries[(maturity, k)] = lsmc.exercise_boundary
            records.append(
                {
                    "ticker": ticker,
                    "spot": s0,
                    "strike": k,
                    "maturity_years": maturity,
                    "volatility": sigma,
                    "risk_free": r,
                    "european_put_bs": euro_put,
                    "american_put_binomial": binomial_put,
                    "lsmc_price": lsmc.price,
                    "lsmc_stderr": lsmc.stderr,
                    "tvr_price": tvr.price,
                    "tvr_stderr": tvr.stderr,
                    "lsmc_minus_binomial": lsmc.price - binomial_put,
                    "tvr_minus_binomial": tvr.price - binomial_put,
                    "binomial_early_exercise_premium": binomial_put - euro_put,
                    "lsmc_early_exercise_premium": lsmc.price - euro_put,
                    "tvr_early_exercise_premium": tvr.price - euro_put,
                }
            )

    out_df = pd.DataFrame(records)
    out_df.to_csv(data_dir / "pricing_results.csv", index=False)

    plot_historical_price(prices, ticker=ticker, output_dir=visuals_dir)
    plot_exercise_boundaries(boundaries=boundaries, steps=steps, output_dir=visuals_dir)
    plot_model_comparison(out_df=out_df, output_dir=visuals_dir)

    display_cols = [
        "strike",
        "maturity_years",
        "european_put_bs",
        "american_put_binomial",
        "lsmc_price",
        "tvr_price",
        "lsmc_minus_binomial",
        "tvr_minus_binomial",
    ]
    print(out_df[display_cols].round(6).to_string(index=False))


if __name__ == "__main__":
    main()
