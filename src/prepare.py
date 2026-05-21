#!/usr/bin/env python
"""
Prepare stage for DVC pipeline: load raw UCI Adult CSV, clean missing
values, and write a processed CSV with headers. Designed for production use
inside DVC pipelines, Docker containers, and experiment runs.

Usage:
    python src/prepare.py --input data/adult.csv --output data/processed.csv

Behavior:
    - Treats '?' as missing values (na_values)
    - Does NOT modify the raw input file
    - Saves cleaned CSV with header row included
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("prepare")


class PrepareConfig:
    """Configuration constants for the prepare stage."""
    COLUMN_NAMES: List[str] = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ]


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare stage: clean raw UCI Adult CSV and produce processed CSV"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw input CSV (headerless), e.g. data/adult.csv",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path where processed CSV will be written, e.g. data/processed.csv",
    )
    parser.add_argument(
        "--na-value",
        default="?",
        help="Value to treat as missing in the raw CSV (default: '?')",
    )
    return parser.parse_args(argv)


def load_raw_csv(path: Path, na_value: str) -> pd.DataFrame:
    """Load a headerless CSV and assign the standard UCI Adult column names.

    Args:
        path: Path to raw CSV
        na_value: String token representing missing values in the raw CSV

    Returns:
        DataFrame with assigned column names
    """
    logger.info("Loading raw dataset: %s", path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    # Use pandas to load headerless CSV; skipinitialspace helps with ' ,'
    df = pd.read_csv(
        path,
        header=None,
        names=PrepareConfig.COLUMN_NAMES,
        skipinitialspace=True,
        na_values=[na_value],
        keep_default_na=True,
    )

    logger.info("Loaded raw CSV with shape: %s", df.shape)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset by removing rows with missing values.

    The function intentionally drops rows with any NA values so downstream
    stages get consistent, fully-observed training data. This keeps the raw
    dataset immutable and preserves reproducibility.
    """
    total_rows = len(df)
    missing_summary = df.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]

    if not missing_cols.empty:
        logger.info("Detected missing values in columns:")
        for col, cnt in missing_cols.items():
            logger.info("  %s: %d (%.2f%%)", col, int(cnt), (cnt / total_rows) * 100)
    else:
        logger.info("No missing values detected in raw data")

    df_clean = df.dropna()
    removed = total_rows - len(df_clean)
    logger.info("Dropped %d rows with missing values; remaining %d rows", removed, len(df_clean))

    if df_clean.empty:
        raise ValueError("All rows were removed after cleaning; no data remains.")

    return df_clean


def validate_schema(df: pd.DataFrame) -> None:
    """Validate the dataframe has the expected number of columns and names."""
    expected = PrepareConfig.COLUMN_NAMES
    if list(df.columns) != expected:
        raise ValueError(f"Schema mismatch: expected columns {expected}, got {list(df.columns)}")
    logger.info("Schema validated: %d columns", len(df.columns))


def save_processed_csv(df: pd.DataFrame, out_path: Path) -> None:
    """Save the cleaned DataFrame to CSV including headers.

    Args:
        df: Cleaned dataframe
        out_path: Path to write CSV
    """
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write CSV with header, no index. This is the artifact consumed by downstream stages.
    df.to_csv(out_path, index=False)
    logger.info("Saved processed CSV to: %s (size: %.2f KB)", out_path, out_path.stat().st_size / 1024)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        logger.info("=== PREPARE STAGE START ===")
        df_raw = load_raw_csv(input_path, args.na_value)
        validate_schema(df_raw)

        logger.info("Starting cleaning step")
        df_clean = clean_dataframe(df_raw)

        # Final validation and persistence
        validate_schema(df_clean)
        save_processed_csv(df_clean, output_path)

        logger.info("=== PREPARE STAGE COMPLETED SUCCESSFULLY ===")
        return 0

    except Exception as exc:  # pragma: no cover - surface errors to caller
        logger.exception("Prepare stage failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
# Placeholder - to be implemented
