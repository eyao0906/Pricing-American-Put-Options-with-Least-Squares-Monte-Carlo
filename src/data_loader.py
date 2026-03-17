from pathlib import Path
import yfinance as yf
import numpy as np
import pandas as pd


def fetch_market_data(ticker: str = "AAPL", period: str = "5y", data_dir: str = "data") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = [c[0] for c in df.columns]
    df = df[["Close"]].dropna().rename(columns={"Close": "close"})
    out = Path(data_dir) / f"{ticker.lower()}_prices.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    return df


def estimate_inputs(df: pd.DataFrame, trading_days: int = 252) -> dict:
    log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
    sigma = float(log_ret.std()) * np.sqrt(trading_days)
    s0 = float(df["close"].iloc[-1])

    # Basic proxy for risk-free rate from 13-week T-bill (^IRX) annualized percent.
    irx = yf.download("^IRX", period="1mo", auto_adjust=True, progress=False)
    if hasattr(irx.columns, "nlevels") and irx.columns.nlevels > 1:
        irx.columns = [c[0] for c in irx.columns]
    if len(irx) == 0:
        r = 0.04
    else:
        r = float(irx["Close"].dropna().iloc[-1]) / 100.0

    return {"s0": s0, "sigma": sigma, "r": r}
