#!/usr/bin/env python
"""
Evaluate stage for DVC pipeline: load trained model and test arrays,
generate predictions, compute metrics, and persist metrics JSON.

Usage:
    python src/evaluate.py --model models/model.joblib --data data/features.npz --metrics metrics/scores.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stage: score trained model and save metrics")
    parser.add_argument("--model", required=True, help="Path to trained model joblib file")
    parser.add_argument("--data", required=True, help="Path to features .npz artifact")
    parser.add_argument("--metrics", required=True, help="Path to save metrics JSON")
    return parser.parse_args()


def load_model(path: Path) -> Any:
    logger.info("Loading model: %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def load_test_data(path: Path) -> Dict[str, np.ndarray]:
    logger.info("Loading test data artifact: %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Features artifact not found: {path}")

    artifact = np.load(path, allow_pickle=False)
    required = {"X_test", "y_test"}
    missing = required - set(artifact.files)
    if missing:
        raise KeyError(f"Features artifact missing required arrays: {missing}")

    return {k: artifact[k] for k in artifact.files}


def compute_metrics(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    logger.info("Generating predictions")
    y_pred = model.predict(X_test)

    logger.info("Generating prediction probabilities")
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Loaded model does not support predict_proba; ROC AUC cannot be computed")
    y_proba = model.predict_proba(X_test)

    logger.info("Evaluating metrics")
    accuracy = accuracy_score(y_test, y_pred)

    # Use positive-class probabilities for binary classification.
    if y_proba.ndim != 2 or y_proba.shape[1] < 2:
        raise ValueError(f"Unexpected probability shape for ROC AUC: {y_proba.shape}")
    auc = roc_auc_score(y_test, y_proba[:, 1])
    f1_macro = f1_score(y_test, y_pred, average="macro")

    metrics = {
        "accuracy": float(accuracy),
        "auc": float(auc),
        "f1_macro": float(f1_macro),
    }
    logger.info("Computed metrics: %s", metrics)
    return metrics


def save_metrics(metrics: Dict[str, float], path: Path) -> None:
    logger.info("Saving metrics to: %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved successfully")


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    data_path = Path(args.data)
    metrics_path = Path(args.metrics)

    try:
        logger.info("=== EVALUATE STAGE START ===")

        model = load_model(model_path)
        features = load_test_data(data_path)

        X_test = features.get("X_test")
        y_test = features.get("y_test")
        logger.info("Loaded test arrays: X_test=%s, y_test=%s", getattr(X_test, "shape", None), getattr(y_test, "shape", None))

        metrics = compute_metrics(model, X_test, y_test)
        save_metrics(metrics, metrics_path)

        logger.info("=== EVALUATE STAGE COMPLETED SUCCESSFULLY ===")
        return 0

    except Exception as exc:
        logger.exception("Evaluate stage failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
# Placeholder - to be implemented
