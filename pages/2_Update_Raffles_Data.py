#!/usr/bin/env python3
"""Upload CSV standardizer for updating raffles.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from raffles_update_utils import (
    DataValidationError,
    build_levels_from_base100,
    build_levels_from_returns,
    parse_and_map_columns,
    save_with_backup,
    standardize_dates,
    standardize_values,
    to_canonical_csv_df,
)

RAFFLES_CSV = Path("raffles.csv")
BACKUP_DIR = Path("backups")
CANONICAL_START = "2016-01-01"


def _process_upload(upload_df: pd.DataFrame, data_type: str) -> dict[str, object]:
    mapped = parse_and_map_columns(upload_df)
    dates = standardize_dates(mapped, "DATE_RAW")
    values = standardize_values(mapped, "VALUE_RAW")

    normalized = pd.DataFrame(
        {
            "__source_row__": mapped["__source_row__"].astype(int),
            "date": dates,
            "value": values,
        }
    )

    duplicate_rows_collapsed = int(
        normalized.duplicated(subset=["date"], keep="last").sum()
    )
    normalized = normalized.sort_values("__source_row__").drop_duplicates(
        subset=["date"], keep="last"
    )

    upload_max = normalized["date"].max()
    start_ts = pd.Timestamp(CANONICAL_START)
    if upload_max < start_ts:
        raise DataValidationError(
            f"Upload max date {upload_max.strftime('%Y-%m-%d')} is before canonical start {CANONICAL_START}."
        )

    points = pd.Series(
        normalized["value"].to_numpy(dtype="float64"),
        index=normalized["date"],
        name="Raffles",
    )

    end_date = upload_max.strftime("%Y-%m-%d")
    if data_type == "Base-100 Levels":
        levels = build_levels_from_base100(points, start_date=CANONICAL_START, end_date=end_date)
    elif data_type == "Monthly Return %":
        levels = build_levels_from_returns(
            points, start_date=CANONICAL_START, end_date=end_date, base_value=100.0
        )
    else:
        raise DataValidationError(f"Unsupported data type: {data_type}")

    out_df = to_canonical_csv_df(levels)
    return {
        "csv_df": out_df,
        "rows": len(out_df),
        "first_date": out_df["YEAR-MONTH"].iloc[0],
        "last_date": out_df["YEAR-MONTH"].iloc[-1],
        "duplicates_collapsed": duplicate_rows_collapsed,
        "invalid_rows": 0,
        "ignored_columns": mapped.attrs.get("ignored_columns", []),
        "data_type": data_type,
    }


def main() -> None:
    st.set_page_config(page_title="Update Raffles Data", layout="wide")
    st.title("Update Raffles Data")
    st.caption(
        "Upload a CSV, standardize it, preview, then save. "
        "Cloud file writes can be ephemeral, so the download is your persistent copy."
    )
    st.write(f"Target file: `{RAFFLES_CSV.resolve()}`")
    st.write(f"Canonical rules: start at `{CANONICAL_START}`, end at uploaded max month, forward-fill gaps.")
    st.write("Expected upload: `DATE` + value column (`Raffles` / `Value` / `Return %` aliases are supported).")

    if "upload_standardize_payload" not in st.session_state:
        st.session_state["upload_standardize_payload"] = None
    if "upload_error_rows" not in st.session_state:
        st.session_state["upload_error_rows"] = 0
    if "upload_last_sig" not in st.session_state:
        st.session_state["upload_last_sig"] = None

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    data_type = st.radio(
        "Data type in uploaded value column",
        options=["Base-100 Levels", "Monthly Return %"],
        horizontal=True,
    )

    if uploaded_file is None:
        st.info("Upload a CSV to start standardization.")
        return

    file_sig = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state["upload_last_sig"] != file_sig:
        st.session_state["upload_last_sig"] = file_sig
        st.session_state["upload_standardize_payload"] = None
        st.session_state["upload_error_rows"] = 0

    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as exc:  # pragma: no cover - UI error path
        st.error(f"Unable to read uploaded CSV: {exc}")
        return

    st.write("Detected columns:", ", ".join(str(c) for c in raw_df.columns))

    c1, c2 = st.columns([1, 1])
    preview_clicked = c1.button("Preview Standardized CSV", type="primary")
    save_clicked = c2.button("Save to raffles.csv")

    if preview_clicked or save_clicked:
        try:
            payload = _process_upload(raw_df, data_type)
            st.session_state["upload_standardize_payload"] = payload
            st.session_state["upload_error_rows"] = 0
        except DataValidationError as exc:
            st.session_state["upload_standardize_payload"] = None
            st.session_state["upload_error_rows"] = len(exc.row_numbers)
            st.error(str(exc))
            return
        except Exception as exc:  # pragma: no cover - UI error path
            st.session_state["upload_standardize_payload"] = None
            st.session_state["upload_error_rows"] = 0
            st.error(str(exc))
            return

    payload = st.session_state.get("upload_standardize_payload")
    if payload is None:
        return

    if payload["ignored_columns"]:
        st.warning(f"Ignored extra columns: {', '.join(str(c) for c in payload['ignored_columns'])}")
    if payload["duplicates_collapsed"] > 0:
        st.warning(
            f"Duplicate dates detected and collapsed (keep-last): {payload['duplicates_collapsed']} row(s)."
        )

    st.subheader("Preview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows in final series", int(payload["rows"]))
    m2.metric("Date range", f"{payload['first_date']} to {payload['last_date']}")
    m3.metric("Duplicate rows collapsed", int(payload["duplicates_collapsed"]))
    m4.metric("Invalid rows", int(st.session_state.get("upload_error_rows", 0)))
    st.write(f"Preview mode: **{payload['data_type']}**")

    preview_df = payload["csv_df"]
    st.dataframe(preview_df, use_container_width=True, height=430)

    csv_bytes = preview_df.to_csv(index=False, float_format="%.2f").encode("utf-8")
    if save_clicked:
        try:
            backup_path = save_with_backup(preview_df, target_csv=RAFFLES_CSV, backup_dir=BACKUP_DIR)
            st.session_state.pop("report_payload", None)
            st.success(f"Saved `{RAFFLES_CSV}`. Backup created at `{backup_path}`.")
        except Exception as exc:  # pragma: no cover - UI error path
            st.error(str(exc))
            return

    st.download_button(
        "Download cleaned raffles.csv",
        data=csv_bytes,
        file_name="raffles.csv",
        mime="text/csv",
        on_click="ignore",
    )


if __name__ == "__main__":
    main()
