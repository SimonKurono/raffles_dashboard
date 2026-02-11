#!/usr/bin/env python3
"""Streamlit app to build/export Raffles benchmark CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF

from build_raffles_benchmark_csv import fetch_rebased_ticker, load_raffles_series, month_start_range

RAFFLES_CSV = Path("raffles.csv")
MAX_TICKERS = 5
DEFAULT_TICKER_ARRAY = "TLT,LQD,JNK,EMB,EMLC"
DEFAULT_DECIMAL_PLACES = 2
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


def prepare_report_tables(
    result: pd.DataFrame, decimal_places: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    levels = result.copy()
    levels["YEAR-MONTH"] = pd.to_datetime(levels["YEAR-MONTH"], errors="coerce")
    levels = levels.set_index("YEAR-MONTH")

    monthly_returns = levels.pct_change() * 100.0
    monthly_returns = monthly_returns.dropna(how="all")

    quarterly_levels = levels.resample("Q").last()
    quarterly_returns = quarterly_levels.pct_change() * 100.0
    quarterly_returns = quarterly_returns.dropna(how="all")
    quarterly_returns.index = quarterly_returns.index.to_period("Q").astype(str)

    yearly_levels = levels.resample("Y").last()
    yearly_returns = yearly_levels.pct_change() * 100.0
    yearly_returns = yearly_returns.dropna(how="all")
    yearly_returns.index = yearly_returns.index.year.astype(str)

    window = 12 if len(levels) >= 12 else len(levels)
    trailing_levels = levels.tail(window)

    month_count = len(levels) - 1
    if month_count > 0:
        cagr = (levels.iloc[-1] / levels.iloc[0]) ** (12.0 / month_count) - 1.0
    else:
        cagr = pd.Series(np.nan, index=levels.columns)

    stats = pd.DataFrame(index=levels.columns)
    stats["Mean Monthly Return (%)"] = monthly_returns.mean()
    stats["Std Monthly Return (%)"] = monthly_returns.std(ddof=1)
    stats["Var Monthly Return"] = monthly_returns.var(ddof=1)
    stats["Annualized Volatility (%)"] = monthly_returns.std(ddof=1) * np.sqrt(12.0)
    stats["Cumulative Return (%)"] = (levels.iloc[-1] / levels.iloc[0] - 1.0) * 100.0
    stats["CAGR (%)"] = cagr * 100.0
    stats["52-Week High (Rebased)"] = trailing_levels.max()
    stats["52-Week Low (Rebased)"] = trailing_levels.min()

    return (
        monthly_returns.round(decimal_places),
        quarterly_returns.round(decimal_places),
        yearly_returns.round(decimal_places),
        stats.round(decimal_places),
    )


def table_with_index_column(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out.index = out.index.strftime("%Y-%m-%d")
    return out.reset_index().rename(columns={"index": index_name})


def format_table_for_pdf(df: pd.DataFrame, index_name: str, decimal_places: int) -> pd.DataFrame:
    table = table_with_index_column(df, index_name)
    for col in table.columns[1:]:
        table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.{decimal_places}f}")
    return table


def add_pdf_table(pdf: FPDF, title: str, table: pd.DataFrame) -> None:
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, title, ln=1)

    if table.empty:
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 6, "No data for selected range.", ln=1)
        pdf.ln(2)
        return

    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    col_count = len(table.columns)
    first_col_width = min(42.0, usable_width * 0.24)
    if col_count == 1:
        widths = [usable_width]
    else:
        other_width = (usable_width - first_col_width) / (col_count - 1)
        widths = [first_col_width] + [other_width] * (col_count - 1)

    row_height = 6.0

    def draw_header() -> None:
        pdf.set_font("Helvetica", "B", 8)
        for i, col in enumerate(table.columns):
            header = str(col)
            if len(header) > 24:
                header = header[:21] + "..."
            pdf.cell(widths[i], row_height, header, border=1, align="C")
        pdf.ln(row_height)
        pdf.set_font("Helvetica", "", 8)

    draw_header()

    for _, row in table.iterrows():
        if pdf.get_y() > (pdf.h - pdf.b_margin - row_height):
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, f"{title} (cont.)", ln=1)
            draw_header()

        for i, value in enumerate(row):
            text = str(value)
            if len(text) > 26:
                text = text[:23] + "..."
            align = "L" if i == 0 else "R"
            pdf.cell(widths[i], row_height, text, border=1, align=align)
        pdf.ln(row_height)

    pdf.ln(2)


def build_pdf_report(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    tickers: list[str],
    decimal_places: int,
    monthly_returns: pd.DataFrame,
    quarterly_returns: pd.DataFrame,
    yearly_returns: pd.DataFrame,
    stats: pd.DataFrame,
) -> bytes:
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 10, "Raffles Benchmark Report", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0,
        6,
        (
            f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} | "
            f"Tickers: {', '.join(tickers)} | Decimals: {decimal_places}"
        ),
        ln=1,
    )
    pdf.ln(2)

    stats_pdf = format_table_for_pdf(
        stats.rename(
            columns={
                "Mean Monthly Return (%)": "Mean Mo Ret %",
                "Std Monthly Return (%)": "Std Mo Ret %",
                "Var Monthly Return": "Var Mo Ret",
                "Annualized Volatility (%)": "Ann Vol %",
                "Cumulative Return (%)": "Cum Ret %",
                "CAGR (%)": "CAGR %",
                "52-Week High (Rebased)": "52W High",
                "52-Week Low (Rebased)": "52W Low",
            }
        ),
        index_name="Series",
        decimal_places=decimal_places,
    )
    yearly_pdf = format_table_for_pdf(yearly_returns, index_name="Year", decimal_places=decimal_places)
    quarterly_pdf = format_table_for_pdf(quarterly_returns, index_name="Quarter", decimal_places=decimal_places)
    monthly_pdf = format_table_for_pdf(monthly_returns, index_name="YEAR-MONTH", decimal_places=decimal_places)

    add_pdf_table(pdf, "Stock Stats", stats_pdf)
    add_pdf_table(pdf, "Yearly Returns (%)", yearly_pdf)
    add_pdf_table(pdf, "Quarterly Returns (%)", quarterly_pdf)
    add_pdf_table(pdf, "Monthly Returns (%)", monthly_pdf)

    raw_pdf = pdf.output(dest="S")
    if isinstance(raw_pdf, (bytes, bytearray)):
        return bytes(raw_pdf)
    return raw_pdf.encode("latin-1")


def build_output(start_date: str, end_date: str, tickers: list[str], decimal_places: int) -> pd.DataFrame:
    monthly_index = month_start_range(start_date, end_date)
    out = pd.DataFrame(index=monthly_index)
    out["Raffles"] = load_raffles_series(RAFFLES_CSV, monthly_index)

    for ticker in tickers:
        out[ticker] = fetch_rebased_ticker(ticker, monthly_index)

    out = out.round(decimal_places)
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

        decimal_places = st.number_input(
            "Decimal places",
            min_value=0,
            max_value=8,
            value=DEFAULT_DECIMAL_PLACES,
            step=1,
            help="Applies to preview and downloaded CSV.",
        )

        submitted = st.form_submit_button("Build CSV")

    if "report_payload" not in st.session_state:
        st.session_state["report_payload"] = None

    if submitted:
        if start_date > end_date:
            st.error("Start date must be on or before end date.")
            return

        parsed_tickers = parse_ticker_array(ticker_array_text)
        if not parsed_tickers:
            st.error("Provide at least one ticker in the ticker array.")
            return

        if len(parsed_tickers) > MAX_TICKERS:
            st.error(f"Provide at most {MAX_TICKERS} unique tickers.")
            return

        chosen = parsed_tickers
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        decimals = int(decimal_places)

        try:
            with st.spinner("Fetching monthly benchmark data from yfinance..."):
                result = build_output(
                    start_date=start_ts.strftime("%Y-%m-%d"),
                    end_date=end_ts.strftime("%Y-%m-%d"),
                    tickers=chosen,
                    decimal_places=decimals,
                )
        except Exception as exc:  # pragma: no cover - UI error path
            st.error(str(exc))
            return

        monthly_returns, quarterly_returns, yearly_returns, stats = prepare_report_tables(
            result=result,
            decimal_places=decimals,
        )

        float_format = f"%.{decimals}f"
        csv_bytes = result.to_csv(index=False, float_format=float_format).encode("utf-8")
        csv_filename = (
            f"raffles_benchmark_{start_ts.strftime('%Y%m%d')}"
            f"_{end_ts.strftime('%Y%m%d')}.csv"
        )

        pdf_bytes = build_pdf_report(
            start_date=start_ts,
            end_date=end_ts,
            tickers=chosen,
            decimal_places=decimals,
            monthly_returns=monthly_returns,
            quarterly_returns=quarterly_returns,
            yearly_returns=yearly_returns,
            stats=stats,
        )
        pdf_filename = (
            f"raffles_benchmark_report_{start_ts.strftime('%Y%m%d')}"
            f"_{end_ts.strftime('%Y%m%d')}.pdf"
        )

        st.session_state["report_payload"] = {
            "tickers": chosen,
            "result": result,
            "monthly_returns": monthly_returns,
            "quarterly_returns": quarterly_returns,
            "yearly_returns": yearly_returns,
            "stats": stats,
            "csv_bytes": csv_bytes,
            "csv_filename": csv_filename,
            "pdf_bytes": pdf_bytes,
            "pdf_filename": pdf_filename,
        }

    payload = st.session_state.get("report_payload")
    if payload is None:
        return

    st.write("Tickers included:", ", ".join(payload["tickers"]))
    st.success(f"Built {len(payload['result'])} monthly rows.")
    st.dataframe(payload["result"], use_container_width=True, height=420)

    st.subheader("Performance Summary")
    tabs = st.tabs(
        [
            "Monthly Returns (%)",
            "Quarterly Returns (%)",
            "Yearly Returns (%)",
            "Stock Stats",
        ]
    )

    with tabs[0]:
        st.dataframe(
            table_with_index_column(payload["monthly_returns"], "YEAR-MONTH"),
            use_container_width=True,
            height=360,
        )
    with tabs[1]:
        st.dataframe(
            table_with_index_column(payload["quarterly_returns"], "Quarter"),
            use_container_width=True,
            height=360,
        )
    with tabs[2]:
        st.dataframe(
            table_with_index_column(payload["yearly_returns"], "Year"),
            use_container_width=True,
            height=360,
        )
    with tabs[3]:
        st.dataframe(
            table_with_index_column(payload["stats"], "Series"),
            use_container_width=True,
            height=360,
        )

    st.download_button(
        "Download CSV",
        data=payload["csv_bytes"],
        file_name=payload["csv_filename"],
        mime="text/csv",
        on_click="ignore",
    )

    st.download_button(
        "Download PDF Report",
        data=payload["pdf_bytes"],
        file_name=payload["pdf_filename"],
        mime="application/pdf",
        on_click="ignore",
    )


if __name__ == "__main__":
    main()
