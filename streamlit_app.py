#!/usr/bin/env python3
"""Streamlit app to build/export Raffles benchmark CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from build_raffles_benchmark_csv import fetch_rebased_ticker, load_raffles_series, month_start_range

RAFFLES_CSV = Path("raffles.csv")
MAX_TICKERS = 5
DEFAULT_TICKER_ARRAY = "TLT,LQD,JNK,EMB,EMLC"
DEFAULT_NUM_TICKERS = 2
EARLIEST_ALLOWED_START = pd.Timestamp("2016-01-01")


def parse_ticker_array(raw: str) -> list[str]:
    tickers: list[str] = []
    seen: set[str] = set()
    for token in raw.replace(" ", "").split(","):
        ticker = token.upper()
        if not ticker:
            continue
        if ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)
    return tickers


def get_raffles_bounds(csv_path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found.")

    df = pd.read_csv(csv_path)
    required = {"YEAR-MONTH", "Raffles"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} must include columns: {sorted(required)}")

    dates = pd.to_datetime(df["YEAR-MONTH"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    values = pd.to_numeric(df["Raffles"], errors="coerce")
    valid_dates = dates[values.notna()]
    if valid_dates.empty:
        raise ValueError("Raffles column has no numeric values.")

    lower = max(valid_dates.min(), EARLIEST_ALLOWED_START)
    upper = valid_dates.max()
    if lower > upper:
        raise ValueError("Invalid date range in raffles.csv.")
    return lower, upper


def build_output(start_date: str, end_date: str, tickers: list[str]) -> pd.DataFrame:
    monthly_index = month_start_range(start_date, end_date)
    out = pd.DataFrame(index=monthly_index)
    out["Raffles"] = load_raffles_series(RAFFLES_CSV, monthly_index)

    for ticker in tickers:
        out[ticker] = fetch_rebased_ticker(ticker, monthly_index)

    out = out.round(2)
    out.insert(0, "YEAR-MONTH", out.index.strftime("%Y-%m-%d"))
    return out


def main() -> None:
    st.set_page_config(page_title="Raffles Benchmark Export", layout="wide")
    st.title("Raffles Benchmark CSV Export")

    try:
        raffles_min, raffles_max = get_raffles_bounds(RAFFLES_CSV)
    except Exception as exc:  # pragma: no cover - UI error path
        st.error(str(exc))
        st.stop()

    st.caption(
        f"Raffles data available from `{raffles_min.strftime('%Y-%m-%d')}` to `{raffles_max.strftime('%Y-%m-%d')}`."
    )

    with st.form("benchmark_form"):
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input(
                "Start date",
                value=raffles_min.date(),
                min_value=raffles_min.date(),
                max_value=raffles_max.date(),
            )
        with c2:
            end_date = st.date_input(
                "End date",
                value=raffles_max.date(),
                min_value=raffles_min.date(),
                max_value=raffles_max.date(),
            )

        ticker_array_text = st.text_input(
            "Ticker array (comma-separated, order matters)",
            value=DEFAULT_TICKER_ARRAY,
            help="Example: TLT,LQD,JNK,EMB,EMLC",
        )

        num_tickers = st.number_input(
            "How many tickers to include (1-5, from left to right in the array)",
            min_value=1,
            max_value=MAX_TICKERS,
            value=DEFAULT_NUM_TICKERS,
            step=1,
        )

        submitted = st.form_submit_button("Build CSV")

    if not submitted:
        return

    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        return

    parsed_tickers = parse_ticker_array(ticker_array_text)
    if not parsed_tickers:
        st.error("Provide at least one ticker in the ticker array.")
        return

    if num_tickers > len(parsed_tickers):
        st.error(
            f"You requested {num_tickers} ticker(s), but only {len(parsed_tickers)} unique ticker(s) were provided."
        )
        return

    chosen = parsed_tickers[:num_tickers]
    st.write("Tickers included:", ", ".join(chosen))

    try:
        with st.spinner("Fetching monthly benchmark data from yfinance..."):
            result = build_output(
                start_date=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
                end_date=pd.Timestamp(end_date).strftime("%Y-%m-%d"),
                tickers=chosen,
            )
    except Exception as exc:  # pragma: no cover - UI error path
        st.error(str(exc))
        return

    st.success(f"Built {len(result)} monthly rows.")
    st.dataframe(result, use_container_width=True, height=420)

    csv_bytes = result.to_csv(index=False, float_format="%.2f").encode("utf-8")
    filename = (
        f"raffles_benchmark_{pd.Timestamp(start_date).strftime('%Y%m%d')}"
        f"_{pd.Timestamp(end_date).strftime('%Y%m%d')}.csv"
    )
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
