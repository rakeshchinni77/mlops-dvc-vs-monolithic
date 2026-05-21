#!/usr/bin/env python
"""
Production-Grade Monolithic ML Training Script
UCI Adult Income Dataset Classification

This script implements a complete, self-contained machine learning workflow
without using DVC orchestration. It serves as a baseline for comparison with
the modular DVC pipeline approach.

Usage:
    python train_monolithic.py

Outputs:
    - model.joblib: Trained RandomForest model
    - metrics.json: Evaluation metrics (accuracy, auc, f1_macro)
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import joblib

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION & SCHEMA
# ============================================================================

class Config:
    """Configuration constants for the training pipeline."""
    
    # Paths
    DATA_PATH = "data/adult.csv"
    MODEL_PATH = "model.joblib"
    METRICS_PATH = "metrics.json"
    
    # Dataset schema (UCI Adult - 15 columns)
    COLUMN_NAMES = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]
    
    # Feature types
    NUMERIC_FEATURES = [
        "age", "fnlwgt", "education_num", 
        "capital_gain", "capital_loss", "hours_per_week"
    ]
    
    CATEGORICAL_FEATURES = [
        "workclass", "education", "marital_status", 
        "occupation", "relationship", "race", "sex", "native_country"
    ]
    
    TARGET = "income"
    
    # Training parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_ESTIMATORS = 100
    MAX_DEPTH = 10


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load UCI Adult Income dataset without headers.
    
    The raw CSV file contains no header row. Column names are assigned
    dynamically using the standard UCI Adult schema.
    
    Args:
        filepath: Path to raw adult.csv file
        
    Returns:
        DataFrame with assigned column names
        
    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If column count is incorrect
    """
    logger.info(f"Loading dataset from: {filepath}")
    
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    
    try:
        df = pd.read_csv(
            filepath,
            header=None,
            names=Config.COLUMN_NAMES,
            skipinitialspace=True,
            na_values=['?']
        )
        
        logger.info(f"✓ Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
        
        # Validate schema
        if df.shape[1] != 15:
            raise ValueError(f"Expected 15 columns, got {df.shape[1]}")
        
        return df
        
    except Exception as e:
        logger.error(f"✗ Failed to load dataset: {str(e)}")
        raise


def inspect_dataset(df: pd.DataFrame) -> None:
    """
    Log dataset inspection details.
    
    Args:
        df: Dataset to inspect
    """
    logger.info("\n" + "=" * 70)
    logger.info("DATASET INSPECTION")
    logger.info("=" * 70)
    
    # Basic info
    logger.info(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.info(f"\nMissing values detected:")
        for col, count in missing[missing > 0].items():
            pct = (count / len(df)) * 100
            logger.info(f"  {col:20s}: {count:6,} ({pct:5.2f}%)")
    else:
        logger.info("No missing values")
    
    # Data types
    logger.info(f"\nData types:")
    for col, dtype in zip(df.columns, df.dtypes):
        logger.info(f"  {col:20s}: {dtype}")
    
    # Target variable
    logger.info(f"\nTarget variable distribution:")
    for class_val, count in df[Config.TARGET].value_counts().items():
        pct = (count / len(df)) * 100
        logger.info(f"  {class_val:10s}: {count:6,} ({pct:5.2f}%)")
    
    logger.info("=" * 70 + "\n")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset by removing rows with missing values.
    
    Args:
        df: Raw dataset
        
    Returns:
        Cleaned dataset
    """
    logger.info("Cleaning dataset...")
    
    initial_rows = len(df)
    df_clean = df.dropna()
    removed_rows = initial_rows - len(df_clean)
    
    logger.info(f"✓ Removed {removed_rows:,} rows with missing values")
    logger.info(f"  Remaining: {len(df_clean):,} rows")
    
    return df_clean


def encode_features(
    df: pd.DataFrame,
    label_encoders: Dict[str, LabelEncoder] = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        df: Dataset to encode
        label_encoders: Dictionary of fitted LabelEncoders (for inference)
        fit: Whether to fit new encoders or use provided ones
        
    Returns:
        Tuple of (encoded_df, label_encoders_dict)
    """
    logger.info("Encoding categorical features...")
    
    df_encoded = df.copy()
    encoders = label_encoders if label_encoders else {}
    
    for col in Config.CATEGORICAL_FEATURES:
        if col not in df_encoded.columns:
            continue
        
        if fit:
            encoders[col] = LabelEncoder()
            df_encoded[col] = encoders[col].fit_transform(
                df_encoded[col].astype(str)
            )
            logger.info(f"✓ Encoded {col}")
        else:
            df_encoded[col] = encoders[col].transform(
                df_encoded[col].astype(str)
            )
    
    return df_encoded, encoders


def encode_target(
    y: pd.Series,
    target_encoder: LabelEncoder = None,
    fit: bool = True
) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode target variable using LabelEncoder.
    
    Args:
        y: Target variable
        target_encoder: Fitted encoder (for inference)
        fit: Whether to fit new encoder
        
    Returns:
        Tuple of (encoded_target, encoder)
    """
    if fit:
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y.astype(str))
        logger.info(f"✓ Encoded target variable")
        logger.info(f"  Classes: {encoder.classes_.tolist()}")
    else:
        y_encoded = target_encoder.transform(y.astype(str))
    
    return y_encoded, target_encoder if not fit else encoder


# ============================================================================
# MODEL TRAINING & EVALUATION
# ============================================================================

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train RandomForest classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained model
    """
    logger.info("\nTraining RandomForestClassifier...")
    logger.info(f"  n_estimators: {Config.N_ESTIMATORS}")
    logger.info(f"  max_depth: {Config.MAX_DEPTH}")
    logger.info(f"  random_state: {Config.RANDOM_STATE}")
    
    model = RandomForestClassifier(
        n_estimators=Config.N_ESTIMATORS,
        max_depth=Config.MAX_DEPTH,
        random_state=Config.RANDOM_STATE,
        n_jobs=-1,  # Use all available cores
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    logger.info(f"✓ Model trained successfully")
    
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Calculates:
    - Accuracy
    - ROC AUC (for binary classification)
    - F1 Macro (weighted average of F1 scores)
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary of metrics
    """
    logger.info("\nEvaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    metrics = {
        "accuracy": float(accuracy),
        "auc": float(auc),
        "f1_macro": float(f1_macro)
    }
    
    logger.info(f"✓ Evaluation complete:")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  AUC-ROC:   {auc:.4f}")
    logger.info(f"  F1 Macro:  {f1_macro:.4f}")
    
    return metrics


def save_model(model: RandomForestClassifier, filepath: str) -> None:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        filepath: Output path
    """
    logger.info(f"\nSaving model to: {filepath}")
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    
    file_size = Path(filepath).stat().st_size / (1024 * 1024)
    logger.info(f"✓ Model saved ({file_size:.2f} MB)")


def save_metrics(metrics: Dict[str, float], filepath: str) -> None:
    """
    Save evaluation metrics to JSON file.
    
    Args:
        metrics: Metrics dictionary
        filepath: Output path
    """
    logger.info(f"Saving metrics to: {filepath}")
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"✓ Metrics saved")
    logger.info(f"  Content: {json.dumps(metrics, indent=2)}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(
    data_path: str = Config.DATA_PATH,
    model_path: str = Config.MODEL_PATH,
    metrics_path: str = Config.METRICS_PATH
) -> None:
    """
    Complete monolithic ML training pipeline.
    
    Args:
        data_path: Path to input dataset
        model_path: Path to save trained model
        metrics_path: Path to save metrics
    """
    try:
        logger.info("=" * 70)
        logger.info("MONOLITHIC ML TRAINING PIPELINE")
        logger.info("=" * 70)
        
        # ====================================================================
        # 1. LOAD DATASET
        # ====================================================================
        logger.info("\n[STAGE 1] LOAD DATASET")
        df = load_dataset(data_path)
        inspect_dataset(df)
        
        # ====================================================================
        # 2. CLEAN DATA
        # ====================================================================
        logger.info("[STAGE 2] CLEAN DATA")
        df = clean_dataset(df)
        
        # ====================================================================
        # 3. SEPARATE FEATURES & TARGET
        # ====================================================================
        logger.info("\n[STAGE 3] SEPARATE FEATURES & TARGET")
        X = df.drop(columns=[Config.TARGET])
        y = df[Config.TARGET]
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        # ====================================================================
        # 4. ENCODE CATEGORICAL FEATURES
        # ====================================================================
        logger.info("\n[STAGE 4] ENCODE CATEGORICAL FEATURES")
        X_encoded, feature_encoders = encode_features(X, fit=True)
        
        # ====================================================================
        # 5. ENCODE TARGET VARIABLE
        # ====================================================================
        logger.info("\n[STAGE 5] ENCODE TARGET VARIABLE")
        y_encoded, target_encoder = encode_target(y, fit=True)
        
        # ====================================================================
        # 6. SPLIT DATA
        # ====================================================================
        logger.info("\n[STAGE 6] SPLIT DATA")
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded,
            y_encoded,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE,
            stratify=y_encoded
        )
        logger.info(f"Training set: {X_train.shape[0]:,} samples")
        logger.info(f"Test set: {X_test.shape[0]:,} samples")
        
        # ====================================================================
        # 7. TRAIN MODEL
        # ====================================================================
        logger.info("\n[STAGE 7] TRAIN MODEL")
        model = train_model(X_train, y_train)
        
        # ====================================================================
        # 8. EVALUATE MODEL
        # ====================================================================
        logger.info("\n[STAGE 8] EVALUATE MODEL")
        metrics = evaluate_model(model, X_test, y_test)
        
        # ====================================================================
        # 9. SAVE ARTIFACTS
        # ====================================================================
        logger.info("\n[STAGE 9] SAVE ARTIFACTS")
        save_model(model, model_path)
        save_metrics(metrics, metrics_path)
        
        # ====================================================================
        # SUMMARY
        # ====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Model saved:   {model_path}")
        logger.info(f"Metrics saved: {metrics_path}")
        logger.info("=" * 70 + "\n")
        
    except Exception as e:
        logger.error(f"\n✗ PIPELINE FAILED: {str(e)}", exc_info=True)
        raise


# ============================================================================
# CLI INTERFACE
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Monolithic ML Training Pipeline for UCI Adult Income Dataset"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default=Config.DATA_PATH,
        help=f"Path to input dataset (default: {Config.DATA_PATH})"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=Config.MODEL_PATH,
        help=f"Path to save model (default: {Config.MODEL_PATH})"
    )
    
    parser.add_argument(
        "--metrics",
        type=str,
        default=Config.METRICS_PATH,
        help=f"Path to save metrics (default: {Config.METRICS_PATH})"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(
        data_path=args.data,
        model_path=args.model,
        metrics_path=args.metrics
    )
