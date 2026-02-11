#!/usr/bin/env python3
"""Data integrity tests for upload parsing/cleaning/updating pipeline."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

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


class TestRafflesUpdateUtils(unittest.TestCase):
    def test_parse_and_map_columns_accepts_aliases_and_tracks_ignored(self) -> None:
        upload_df = pd.DataFrame(
            {
                "Year-Month": ["2016-01-01", "2016-02-01"],
                "Monthly Return %": [0.0, 2.5],
                "Comment": ["a", "b"],
            }
        )

        mapped = parse_and_map_columns(upload_df)

        self.assertEqual(list(mapped.columns), ["__source_row__", "DATE_RAW", "VALUE_RAW"])
        self.assertEqual(mapped.attrs["date_col"], "Year-Month")
        self.assertEqual(mapped.attrs["value_col"], "Monthly Return %")
        self.assertEqual(mapped.attrs["ignored_columns"], ["Comment"])
        self.assertEqual(mapped["__source_row__"].tolist(), [2, 3])

    def test_standardize_dates_reports_exact_bad_rows(self) -> None:
        df = pd.DataFrame(
            {
                "__source_row__": [2, 3, 4],
                "DATE_RAW": ["2016-01-01", "not-a-date", "2016-03-01"],
            }
        )

        with self.assertRaises(DataValidationError) as ctx:
            standardize_dates(df, "DATE_RAW")

        self.assertEqual(ctx.exception.row_numbers, [3])

    def test_standardize_values_reports_exact_bad_rows(self) -> None:
        df = pd.DataFrame(
            {
                "__source_row__": [2, 3, 4],
                "VALUE_RAW": [1.0, "bad", ""],
            }
        )

        with self.assertRaises(DataValidationError) as ctx:
            standardize_values(df, "VALUE_RAW")

        self.assertEqual(ctx.exception.row_numbers, [3, 4])

    def test_build_levels_from_returns_compounds_and_carries_forward_missing(self) -> None:
        monthly_returns = pd.Series(
            [5.0, -1.0],
            index=pd.to_datetime(["2016-02-01", "2016-03-01"]),
        )

        levels = build_levels_from_returns(
            monthly_returns,
            start_date="2016-01-01",
            end_date="2016-04-01",
            base_value=100.0,
        )

        self.assertAlmostEqual(levels.loc[pd.Timestamp("2016-01-01")], 100.0, places=6)
        self.assertAlmostEqual(levels.loc[pd.Timestamp("2016-02-01")], 105.0, places=6)
        self.assertAlmostEqual(levels.loc[pd.Timestamp("2016-03-01")], 103.95, places=6)
        # Missing Apr return => forward carry.
        self.assertAlmostEqual(levels.loc[pd.Timestamp("2016-04-01")], 103.95, places=6)

    def test_build_levels_from_base100_requires_2016_01_anchor(self) -> None:
        points = pd.Series([110.0], index=pd.to_datetime(["2016-02-01"]))

        with self.assertRaises(DataValidationError):
            build_levels_from_base100(points, start_date="2016-01-01", end_date="2016-03-01")

    def test_build_levels_from_base100_forward_fills_gaps(self) -> None:
        points = pd.Series(
            [100.0, 120.0],
            index=pd.to_datetime(["2016-01-01", "2016-03-01"]),
        )

        levels = build_levels_from_base100(points, start_date="2016-01-01", end_date="2016-04-01")

        self.assertAlmostEqual(levels.loc[pd.Timestamp("2016-01-01")], 100.0, places=6)
        self.assertAlmostEqual(levels.loc[pd.Timestamp("2016-02-01")], 100.0, places=6)
        self.assertAlmostEqual(levels.loc[pd.Timestamp("2016-03-01")], 120.0, places=6)
        self.assertAlmostEqual(levels.loc[pd.Timestamp("2016-04-01")], 120.0, places=6)

    def test_to_canonical_csv_df_formats_shape_and_rounding(self) -> None:
        levels = pd.Series(
            [100.123, 101.126],
            index=pd.to_datetime(["2016-01-01", "2016-02-01"]),
            name="Raffles",
        )

        out = to_canonical_csv_df(levels)

        self.assertEqual(list(out.columns), ["YEAR-MONTH", "Raffles"])
        self.assertEqual(out["YEAR-MONTH"].tolist(), ["2016-01-01", "2016-02-01"])
        self.assertEqual(out["Raffles"].tolist(), [100.12, 101.13])

    def test_save_with_backup_creates_backup_and_updates_target(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            target = tmp / "raffles.csv"
            backup_dir = tmp / "backups"

            # Existing file should be copied into backup before overwrite.
            target.write_text("YEAR-MONTH,Raffles\n2016-01-01,99.00\n", encoding="utf-8")

            new_df = pd.DataFrame(
                {
                    "YEAR-MONTH": ["2016-01-01", "2016-02-01"],
                    "Raffles": [100.0, 101.25],
                }
            )

            backup_path = save_with_backup(new_df, target, backup_dir)
            self.assertTrue(backup_path.exists())
            self.assertTrue(target.exists())

            backup_text = backup_path.read_text(encoding="utf-8")
            target_text = target.read_text(encoding="utf-8")

            self.assertIn("2016-01-01,99.00", backup_text)
            self.assertIn("2016-01-01,100.00", target_text)
            self.assertIn("2016-02-01,101.25", target_text)

    def test_process_upload_keeps_last_duplicate_month(self) -> None:
        page_path = Path("pages/2_Update_Raffles_Data.py")
        spec = importlib.util.spec_from_file_location("update_page", page_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        upload_df = pd.DataFrame(
            {
                "DATE": ["2016-01-01", "2016-02-01", "2016-02-20"],
                "Monthly Return %": [0.0, 1.0, 2.0],
            }
        )

        payload = module._process_upload(upload_df, "Monthly Return %")
        out = payload["csv_df"]

        self.assertEqual(payload["duplicates_collapsed"], 1)
        feb = float(out.loc[out["YEAR-MONTH"] == "2016-02-01", "Raffles"].iloc[0])
        # Keep-last duplicate means Feb return should be 2%, not 1%.
        self.assertAlmostEqual(feb, 102.0, places=6)


if __name__ == "__main__":
    unittest.main()
