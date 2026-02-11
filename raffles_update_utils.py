#!/usr/bin/env python3
"""Upload-driven standardization helpers for raffles.csv."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd


CANONICAL_START = "2016-01-01"


class DataValidationError(ValueError):
    """Validation error that optionally includes exact CSV row numbers."""

    def __init__(self, message: str, row_numbers: list[int] | None = None) -> None:
        super().__init__(message)
        self.row_numbers = row_numbers or []


def _norm_col(name: str) -> str:
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


def _pick_single_column(columns: list[str], aliases: set[str], label: str) -> str:
    matches = [col for col in columns if _norm_col(col) in aliases]
    if not matches:
        raise DataValidationError(
            f"Could not find a {label} column. Detected columns: {columns}."
        )
    if len(matches) > 1:
        raise DataValidationError(
            f"Ambiguous {label} columns {matches}. Keep exactly one {label} column."
        )
    return matches[0]


def parse_and_map_columns(upload_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map flexible CSV headers to canonical DATE_RAW and VALUE_RAW columns.
    Adds __source_row__ for row-precise validation messages (CSV line numbers).
    """
    if upload_df.empty:
        raise DataValidationError("Uploaded CSV is empty.")

    columns = list(upload_df.columns)
    date_aliases = {
        "date",
        "year month",
        "yearmonth",
        "month",
    }
    value_aliases = {
        "raffles",
        "value",
        "return",
        "return %",
        "monthly return",
        "monthly return %",
        "returns",
    }

    date_col = _pick_single_column(columns, date_aliases, "date")
    value_col = _pick_single_column(columns, value_aliases, "value")

    mapped = pd.DataFrame(
        {
            "__source_row__": (upload_df.index + 2).astype(int),
            "DATE_RAW": upload_df[date_col],
            "VALUE_RAW": upload_df[value_col],
        }
    )
    mapped.attrs["date_col"] = date_col
    mapped.attrs["value_col"] = value_col
    mapped.attrs["ignored_columns"] = [c for c in columns if c not in {date_col, value_col}]
    return mapped


def standardize_dates(df: pd.DataFrame, date_col: str) -> pd.Series:
    """Parse dates and normalize to month-start timestamps."""
    if date_col not in df.columns:
        raise DataValidationError(f"Missing date column '{date_col}'.")

    parsed = pd.to_datetime(df[date_col], errors="coerce")
    bad = parsed.isna()
    if bad.any():
        bad_rows = df.loc[bad, "__source_row__"].astype(int).tolist() if "__source_row__" in df.columns else (
            (df.index[bad] + 1).astype(int).tolist()
        )
        raise DataValidationError(
            f"Invalid date values in CSV row(s): {', '.join(str(r) for r in bad_rows)}.",
            row_numbers=bad_rows,
        )

    return parsed.dt.to_period("M").dt.to_timestamp()


def standardize_values(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Parse numeric values and fail fast on invalid rows."""
    if value_col not in df.columns:
        raise DataValidationError(f"Missing value column '{value_col}'.")

    raw = df[value_col]
    parsed = pd.to_numeric(raw, errors="coerce")

    raw_text = raw.astype(str).str.strip()
    invalid = parsed.isna() | raw_text.eq("")
    if invalid.any():
        bad_rows = df.loc[invalid, "__source_row__"].astype(int).tolist() if "__source_row__" in df.columns else (
            (df.index[invalid] + 1).astype(int).tolist()
        )
        raise DataValidationError(
            f"Invalid numeric values in CSV row(s): {', '.join(str(r) for r in bad_rows)}.",
            row_numbers=bad_rows,
        )

    return parsed.astype(float)


def forward_fill_canonical(series: pd.Series) -> pd.Series:
    """Forward fill missing months."""
    return series.ffill()


def _canonical_index(start_date: str, end_date: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start_date, end=end_date, freq="MS")


def build_levels_from_base100(series: pd.Series, start_date: str, end_date: str) -> pd.Series:
    """
    Build canonical level series from base-100 level points.
    Requires a valid anchor value at start_date.
    """
    if end_date < start_date:
        raise DataValidationError("Uploaded maximum date is before canonical start date 2016-01-01.")

    canonical = _canonical_index(start_date, end_date)
    out = series.sort_index().reindex(canonical)

    start_ts = pd.Timestamp(start_date)
    if pd.isna(out.loc[start_ts]):
        raise DataValidationError(
            "Base-100 upload must include a valid value for 2016-01-01."
        )

    out = forward_fill_canonical(out)
    if out.isna().any():
        raise DataValidationError("Unable to forward fill base-100 series. Check upload values.")

    out.name = "Raffles"
    return out.astype(float)


def build_levels_from_returns(
    series: pd.Series,
    start_date: str,
    end_date: str,
    base_value: float = 100.0,
) -> pd.Series:
    """
    Build canonical levels from monthly return percentages.
    levels[t] = levels[t-1] * (1 + ret_pct[t]/100), missing return -> carry forward.
    """
    if end_date < start_date:
        raise DataValidationError("Uploaded maximum date is before canonical start date 2016-01-01.")

    canonical = _canonical_index(start_date, end_date)
    returns = series.sort_index().reindex(canonical)

    levels = pd.Series(index=canonical, dtype="float64", name="Raffles")
    levels.iloc[0] = float(base_value)

    for i in range(1, len(levels)):
        prev = float(levels.iloc[i - 1])
        ret = returns.iloc[i]
        if pd.isna(ret):
            levels.iloc[i] = prev
        else:
            levels.iloc[i] = prev * (1.0 + float(ret) / 100.0)

    return levels


def to_canonical_csv_df(levels: pd.Series) -> pd.DataFrame:
    """Convert standardized levels to canonical raffles.csv shape."""
    df = pd.DataFrame(
        {
            "YEAR-MONTH": pd.to_datetime(levels.index).strftime("%Y-%m-%d"),
            "Raffles": levels.astype(float).round(2),
        }
    )
    return df


def save_with_backup(df: pd.DataFrame, target_csv: Path, backup_dir: Path) -> Path:
    """Backup current file and save canonical raffles.csv."""
    required = {"YEAR-MONTH", "Raffles"}
    if not required.issubset(df.columns):
        raise DataValidationError(f"Save dataframe must include columns: {sorted(required)}")

    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"raffles_{stamp}.csv"

    if target_csv.exists():
        shutil.copy2(target_csv, backup_path)
    else:
        df.loc[:, ["YEAR-MONTH", "Raffles"]].to_csv(backup_path, index=False, float_format="%.2f")

    df.loc[:, ["YEAR-MONTH", "Raffles"]].to_csv(target_csv, index=False, float_format="%.2f")
    return backup_path
