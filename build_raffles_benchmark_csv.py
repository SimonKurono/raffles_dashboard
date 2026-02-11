#!/usr/bin/env python3
"""Build a fully populated monthly Raffles benchmark CSV using yfinance."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# User-editable defaults
DEFAULT_TICKERS = ["TLT", "LQD", "JNK", "EMB", "EMLC"]
DEFAULT_NUM_TICKERS = 2
DEFAULT_INPUT_CSV = Path("raffles.csv")
DEFAULT_OUTPUT_CSV = Path("raffles.csv")
DEFAULT_START_DATE = "2016-01-01"
DEFAULT_END_DATE = "2025-10-01"


def month_start_range(start_date: str, end_date: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start_date, end=end_date, freq="MS")


def _fill_monthly_raffles(values: pd.Series) -> pd.Series:
    """Fill missing months by interpolating in log space (geometric interpolation)."""
    filled = values.astype(float).copy()
    known = filled.dropna()
    if known.empty:
        raise ValueError("Raffles column has no numeric values to anchor interpolation.")

    if (known <= 0).any():
        filled = filled.interpolate(method="time", limit_direction="both")
    else:
        logged = np.log(filled)
        filled = np.exp(logged.interpolate(method="time", limit_direction="both"))

    return filled.ffill().bfill()


def load_raffles_series(csv_path: Path, monthly_index: pd.DatetimeIndex) -> pd.Series:
    df = pd.read_csv(csv_path)
    required = {"YEAR-MONTH", "Raffles"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} must include columns: {sorted(required)}")

    dates = pd.to_datetime(df["YEAR-MONTH"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    raffles_values = pd.to_numeric(df["Raffles"], errors="coerce").to_numpy()
    raffles = pd.Series(raffles_values, index=dates, name="Raffles")
    raffles = raffles.groupby(level=0).last().sort_index()
    raffles = raffles.reindex(monthly_index)

    return _fill_monthly_raffles(raffles)


def fetch_rebased_ticker(ticker: str, monthly_index: pd.DatetimeIndex) -> pd.Series:
    start_date = monthly_index.min().strftime("%Y-%m-%d")
    end_for_download = (monthly_index.max() + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")

    data = yf.download(
        ticker,
        start=start_date,
        end=end_for_download,
        interval="1mo",
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No monthly data returned for ticker '{ticker}'.")

    # yfinance may return either flat columns or a MultiIndex even for one ticker.
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" not in data.columns.get_level_values(0):
            raise ValueError(f"No monthly Adjusted Close data returned for ticker '{ticker}'.")
        adj_close = data.xs("Adj Close", axis=1, level=0)
    else:
        if "Adj Close" not in data.columns:
            raise ValueError(f"No monthly Adjusted Close data returned for ticker '{ticker}'.")
        adj_close = data["Adj Close"]

    if isinstance(adj_close, pd.DataFrame):
        if ticker in adj_close.columns:
            adj_close = adj_close[ticker]
        elif adj_close.shape[1] == 1:
            adj_close = adj_close.iloc[:, 0]
        else:
            raise ValueError(
                f"Adjusted Close extraction for '{ticker}' is ambiguous; received columns {list(adj_close.columns)}."
            )

    prices = pd.to_numeric(adj_close, errors="coerce").astype("float64")
    prices = prices.dropna()
    prices.index = pd.to_datetime(prices.index).to_period("M").to_timestamp()
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    prices = prices.reindex(monthly_index).ffill()

    if prices.isna().any():
        first_missing = prices[prices.isna()].index[0].strftime("%Y-%m-%d")
        raise ValueError(
            f"Ticker '{ticker}' is missing data at the start of the range (first missing month: {first_missing}). "
            "Choose a ticker with full history or adjust the date range."
        )

    monthly_returns = prices.pct_change().fillna(0.0)
    rebased = (1.0 + monthly_returns).cumprod() * 100.0
    rebased.iloc[0] = 100.0
    rebased.name = ticker

    return rebased


def select_tickers(tickers: list[str], num_tickers: int) -> list[str]:
    cleaned = [ticker.strip().upper() for ticker in tickers if ticker and ticker.strip()]
    if not cleaned:
        raise ValueError("Provide at least one ticker.")
    if num_tickers < 1:
        raise ValueError("num_tickers must be at least 1.")
    if num_tickers > len(cleaned):
        raise ValueError(
            f"num_tickers ({num_tickers}) cannot exceed provided ticker count ({len(cleaned)})."
        )
    return cleaned[:num_tickers]


def build_benchmark_csv(
    input_csv: Path,
    output_csv: Path,
    tickers: list[str],
    num_tickers: int,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    monthly_index = month_start_range(start_date, end_date)
    chosen_tickers = select_tickers(tickers, num_tickers)

    final = pd.DataFrame(index=monthly_index)
    final["Raffles"] = load_raffles_series(input_csv, monthly_index)

    for ticker in chosen_tickers:
        final[ticker] = fetch_rebased_ticker(ticker, monthly_index)

    final = final.round(2)
    final.insert(0, "YEAR-MONTH", final.index.strftime("%Y-%m-%d"))
    final.to_csv(output_csv, index=False, float_format="%.2f")
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge monthly yfinance benchmarks into raffles.csv and output a full "
            "monthly series from 2016-01-01 to 2025-10-01."
        )
    )
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Ticker list (space-separated). Example: --tickers TLT LQD JNK",
    )
    parser.add_argument(
        "--num-tickers",
        type=int,
        default=DEFAULT_NUM_TICKERS,
        help="How many tickers from --tickers to include, taking from left to right.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final = build_benchmark_csv(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        tickers=args.tickers,
        num_tickers=args.num_tickers,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    selected = select_tickers(args.tickers, args.num_tickers)
    print(f"Wrote {len(final)} rows to {args.output_csv} with tickers: {', '.join(selected)}")


if __name__ == "__main__":
    main()
