#!/usr/bin/env python
"""
Train stage for DVC pipeline: load features artifact, read training
hyperparameters from `params.yaml`, train RandomForestClassifier, and
serialize the model to disk. Designed for production use inside DVC
pipelines, Docker containers, and automated experiment runs.

Usage:
    python src/train.py --input data/features.npz --output models/model.joblib
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train")


DEFAULT_PARAMS_PATH = Path("params.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stage: train model from features artifact")
    parser.add_argument("--input", required=True, help="Path to features .npz artifact")
    parser.add_argument("--output", required=True, help="Path to save trained model (joblib)")
    parser.add_argument("--params", default=str(DEFAULT_PARAMS_PATH), help="Path to params.yaml")
    return parser.parse_args()


def load_features(path: Path) -> Dict[str, np.ndarray]:
    logger.info("Loading features artifact: %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Features artifact not found: {path}")

    data = np.load(path, allow_pickle=False)
    keys = set(data.files)
    required = {"X_train", "y_train"}
    missing = required - keys
    if missing:
        raise KeyError(f"Features artifact missing required arrays: {missing}")

    # Return a plain dict for ease of use
    return {k: data[k] for k in data.files}


def read_params(path: Path) -> Dict[str, Any]:
    logger.info("Reading params from: %s", path)
    if not path.exists():
        raise FileNotFoundError(f"params.yaml not found at: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    train_cfg = cfg.get("train")
    if train_cfg is None:
        raise KeyError("Missing 'train' section in params.yaml")

    required_keys = ["model_type", "n_estimators", "max_depth", "random_state"]
    missing = [k for k in required_keys if k not in train_cfg]
    if missing:
        raise KeyError(f"Missing required train params: {missing}")

    logger.info("Loaded train params: %s", {k: train_cfg[k] for k in required_keys})
    return train_cfg


def instantiate_model(train_cfg: Dict[str, Any]) -> RandomForestClassifier:
    model_type = str(train_cfg.get("model_type")).lower()
    if model_type not in {"random_forest", "randomforest", "rf", "random-forest"}:
        raise ValueError(f"Unsupported model_type: {model_type}. Supported: 'random_forest'")

    n_estimators = int(train_cfg.get("n_estimators"))
    max_depth = train_cfg.get("max_depth")
    max_depth = None if max_depth is None else int(max_depth)
    random_state = int(train_cfg.get("random_state"))

    logger.info("Instantiating RandomForestClassifier: n_estimators=%s, max_depth=%s, random_state=%s",
                n_estimators, max_depth, random_state)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    return model


def save_model(model: Any, out_path: Path) -> None:
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    logger.info("Saved model to %s (size: %.2f MB)", out_path, out_path.stat().st_size / (1024 * 1024))


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    params_path = Path(args.params)

    try:
        logger.info("=== TRAIN STAGE START ===")

        features = load_features(input_path)
        X_train = features.get("X_train")
        y_train = features.get("y_train")

        logger.info("Feature shapes: X_train=%s, y_train=%s", getattr(X_train, "shape", None), getattr(y_train, "shape", None))

        params = read_params(params_path)
        model = instantiate_model(params)

        logger.info("Starting model.fit()")
        model.fit(X_train, y_train)
        logger.info("Model training completed")

        save_model(model, output_path)

        logger.info("=== TRAIN STAGE COMPLETED SUCCESSFULLY ===")
        return 0

    except Exception as exc:
        logger.exception("Train stage failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
# Placeholder - to be implemented
