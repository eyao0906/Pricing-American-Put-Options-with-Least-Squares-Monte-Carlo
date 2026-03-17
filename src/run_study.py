from pathlib import Path
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_loader import fetch_market_data, estimate_inputs
from lsmc import price_american_put_lsmc
from tsitsiklis_vanroy import price_american_put_tvr


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_european_put(s0: float, k: float, r: float, sigma: float, t: float) -> float:
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return k * math.exp(-r * t) * norm_cdf(-d2) - s0 * norm_cdf(-d1)


def main():
    ticker = "AAPL"
    maturity = 1.0
    steps = 50
    n_paths = 100_000

    prices = fetch_market_data(ticker=ticker, period="5y", data_dir="data")
    inputs = estimate_inputs(prices)
    s0, sigma, r = inputs["s0"], inputs["sigma"], inputs["r"]

    strikes = [0.9 * s0, s0, 1.1 * s0]
    records = []

    boundaries = {}
    for k in strikes:
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
        boundaries[k] = lsmc.exercise_boundary

        euro_put = bs_european_put(s0, k, r, sigma, maturity)
        records.append(
            {
                "ticker": ticker,
                "spot": s0,
                "strike": k,
                "maturity_years": maturity,
                "volatility": sigma,
                "risk_free": r,
                "european_put_bs": euro_put,
                "lsmc_price": lsmc.price,
                "lsmc_stderr": lsmc.stderr,
                "tvr_price": tvr.price,
                "tvr_stderr": tvr.stderr,
                "abs_diff": abs(lsmc.price - tvr.price),
            }
        )

    out_df = pd.DataFrame(records)
    Path("data").mkdir(exist_ok=True)
    out_df.to_csv("data/pricing_results.csv", index=False)

    Path("visuals").mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(prices.index, prices["close"], label=f"{ticker} close")
    plt.title(f"{ticker} historical close (5y)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visuals/historical_price.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    x = np.arange(1, steps + 1)
    for k, b in boundaries.items():
        y = b[1:]
        plt.plot(x, y, label=f"K={k:.2f}")
    plt.title("Estimated LSMC exercise boundary (American put)")
    plt.xlabel("Time step")
    plt.ylabel("Boundary stock price")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visuals/exercise_boundary.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    width = 0.35
    idx = np.arange(len(out_df))
    plt.bar(idx - width / 2, out_df["lsmc_price"], width, label="LSMC")
    plt.bar(idx + width / 2, out_df["tvr_price"], width, label="TVR")
    plt.xticks(idx, [f"K={k:.1f}" for k in out_df["strike"]])
    plt.ylabel("Option price")
    plt.title("American put prices: LSMC vs TVR")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visuals/model_comparison.png", dpi=180)
    plt.close()

    print(out_df)


if __name__ == "__main__":
    main()
