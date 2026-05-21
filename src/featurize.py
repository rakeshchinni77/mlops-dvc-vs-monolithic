#!/usr/bin/env python
"""
Featurize stage for DVC pipeline: load processed CSV, encode categorical
features, split train/test, and save compressed features artifact.

Usage:
    python src/featurize.py --input data/processed.csv --output data/features.npz

This stage is designed to be deterministic and compatible with DVC and Docker.
It reads `params.yaml` for `prepare.test_size` and `prepare.random_state`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("featurize")


DEFAULT_PARAMS_PATH = Path("params.yaml")


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

NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

TARGET = "income"


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Featurize processed CSV into features artifact")
    parser.add_argument("--input", required=True, help="Path to processed CSV (with header)")
    parser.add_argument("--output", required=True, help="Path to save features (.npz)")
    parser.add_argument("--params", default=str(DEFAULT_PARAMS_PATH), help="Path to params.yaml")
    return parser.parse_args(argv)


def load_params(path: Path) -> Tuple[int, float]:
    if not path.exists():
        raise FileNotFoundError(f"params.yaml not found at: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    random_state = int(cfg.get("prepare", {}).get("random_state", 42))
    test_size = float(cfg.get("prepare", {}).get("test_size", 0.2))
    logger.info("Loaded params: prepare.random_state=%s, prepare.test_size=%s", random_state, test_size)
    return random_state, test_size


def load_processed_csv(path: Path) -> pd.DataFrame:
    logger.info("Loading processed CSV: %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Processed CSV not found: {path}")
    df = pd.read_csv(path, header=0, skipinitialspace=True)
    if list(df.columns) != COLUMN_NAMES:
        raise ValueError(f"Processed CSV schema mismatch. Expected columns: {COLUMN_NAMES}")
    logger.info("Loaded processed CSV with shape: %s", df.shape)
    return df


def encode_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Encode categorical features with LabelEncoder and return (X, y).

    The encoders are fitted on the provided dataframe to ensure consistency
    within an experiment run.
    """
    df_copy = df.copy()

    # Encode categorical features
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        enc = LabelEncoder()
        df_copy[col] = enc.fit_transform(df_copy[col].astype(str))
        encoders[col] = enc
        logger.info("Encoded %s (n_unique=%d)", col, df_copy[col].nunique())

    # Numeric features: ensure numeric dtype
    for col in NUMERIC_FEATURES:
        df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")

    # Encode target
    target_enc = LabelEncoder()
    y = target_enc.fit_transform(df_copy[TARGET].astype(str))
    logger.info("Encoded target classes: %s", target_enc.classes_.tolist())

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df_copy[feature_cols].to_numpy(dtype=float)

    return X, y


def split_and_save(X: np.ndarray, y: np.ndarray, output: Path, test_size: float, random_state: int) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    logger.info("Saved features artifact to: %s", output)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)
    params_path = Path(args.params)

    try:
        logger.info("=== FEATURIZE STAGE START ===")
        random_state, test_size = load_params(params_path)
        df = load_processed_csv(input_path)
        X, y = encode_dataframe(df)
        split_and_save(X, y, output_path, test_size=test_size, random_state=random_state)
        logger.info("=== FEATURIZE STAGE COMPLETED SUCCESSFULLY ===")
        return 0
    except Exception as exc:
        logger.exception("Featurize stage failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
# Placeholder - to be implemented
