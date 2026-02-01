#!/usr/bin/env python3
"""
validate_model.py
-----------------------
Model Validation Script for CI/CD Pipeline

This script validates a trained model against quality thresholds before deployment.
It is designed to be run in GitHub Actions or any CI/CD pipeline.

Exit Codes:
- 0: Model passes all validation checks
- 1: Model fails validation (metrics below threshold or errors)

Usage:
    python scripts/validate_model.py [--model-path PATH] [--rmse-threshold N] [--r2-threshold N]

Example:
    python scripts/validate_model.py --rmse-threshold 150000 --r2-threshold 0.75
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from feature_transformer import FeatureTransformer, NUMERIC_FEATURES, CATEGORICAL_FEATURES

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data_lake" / "gold" / "property_prices_csv"
VALIDATION_REPORT_PATH = PROJECT_ROOT / "reports" / "validation_report.json"

# Default thresholds (can be overridden via CLI)
DEFAULT_RMSE_THRESHOLD = 150000  # Maximum acceptable RMSE (€)
DEFAULT_R2_THRESHOLD = 0.70     # Minimum acceptable R²
DEFAULT_MAE_THRESHOLD = 100000  # Maximum acceptable MAE (€)

TARGET = "price"


# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

def load_model_artifacts(model_dir: Path) -> tuple:
    """
    Load the trained model and transformer from disk.
    
    Args:
        model_dir: Path to models directory
        
    Returns:
        Tuple of (model, transformer, metadata)
    """
    model_path = model_dir / "champion_model.pkl"
    transformer_path = model_dir / "feature_transformer.pkl"
    metadata_path = model_dir / "model_metadata.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    if not transformer_path.exists():
        raise FileNotFoundError(f"Transformer not found at {transformer_path}")
    
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    transformer = joblib.load(transformer_path)
    
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, transformer, metadata


def load_validation_data() -> pd.DataFrame:
    """
    Load data for validation. Uses a holdout portion of the gold layer data.
    
    Returns:
        pd.DataFrame: Validation dataset
    """
    csv_folders = sorted([f for f in DATA_PATH.iterdir() if f.is_dir()])
    
    if not csv_folders:
        raise FileNotFoundError(f"No data found in {DATA_PATH}. Run data pipeline first.")
    
    latest_folder = csv_folders[-1]
    print(f"Loading validation data from: {latest_folder}")
    
    csv_files = list(latest_folder.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {latest_folder}")
    
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")
    
    data = pd.concat(dfs, ignore_index=True)
    
    # Use a consistent random sample for validation (20% holdout)
    # Using fixed random_state ensures reproducibility
    validation_data = data.sample(frac=0.2, random_state=42)
    
    print(f"Loaded {len(validation_data):,} validation samples")
    return validation_data


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features and target for validation.
    
    Args:
        df: Raw dataframe
        
    Returns:
        Tuple of (X features DataFrame, y target Series)
    """
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    available_features = [c for c in feature_cols if c in df.columns]
    
    X = df[available_features].copy()
    y = df[TARGET].copy()
    
    # Handle missing values (same as training)
    for col in NUMERIC_FEATURES:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(X[col].median())
    
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna('Unknown').astype(str)
    
    # Remove rows with missing target
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    return X, y


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dict with metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Additional metrics for monitoring
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
    
    # Prediction distribution stats
    residuals = y_true - y_pred
    
    return {
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "mae": round(mae, 2),
        "mape": round(mape, 2),
        "residual_mean": round(np.mean(residuals), 2),
        "residual_std": round(np.std(residuals), 2),
        "n_samples": len(y_true)
    }


def validate_model(
    model,
    transformer: FeatureTransformer,
    X: pd.DataFrame,
    y: pd.Series,
    rmse_threshold: float,
    r2_threshold: float,
    mae_threshold: float
) -> tuple:
    """
    Run model validation against thresholds.
    
    Args:
        model: Trained sklearn model
        transformer: Feature transformer
        X: Feature DataFrame
        y: Target Series
        rmse_threshold: Maximum acceptable RMSE
        r2_threshold: Minimum acceptable R²
        mae_threshold: Maximum acceptable MAE
        
    Returns:
        Tuple of (passed: bool, results: dict)
    """
    print("\n" + "="*60)
    print("🔍 RUNNING MODEL VALIDATION")
    print("="*60)
    
    # Transform features
    print("\nTransforming features...")
    X_transformed = transformer.transform(X)
    
    # Make predictions
    print("Running predictions...")
    y_pred = model.predict(X_transformed)
    
    # Compute metrics
    metrics = compute_metrics(y.values, y_pred)
    
    # Check thresholds
    checks = {
        "rmse_check": {
            "value": metrics["rmse"],
            "threshold": rmse_threshold,
            "passed": metrics["rmse"] <= rmse_threshold,
            "operator": "<="
        },
        "r2_check": {
            "value": metrics["r2"],
            "threshold": r2_threshold,
            "passed": metrics["r2"] >= r2_threshold,
            "operator": ">="
        },
        "mae_check": {
            "value": metrics["mae"],
            "threshold": mae_threshold,
            "passed": metrics["mae"] <= mae_threshold,
            "operator": "<="
        }
    }
    
    # Print results
    print("\n📊 VALIDATION METRICS:")
    print("-" * 40)
    print(f"  RMSE:  €{metrics['rmse']:,.2f}  (threshold: €{rmse_threshold:,.0f})")
    print(f"  R²:    {metrics['r2']:.4f}      (threshold: {r2_threshold})")
    print(f"  MAE:   €{metrics['mae']:,.2f}  (threshold: €{mae_threshold:,.0f})")
    print(f"  MAPE:  {metrics['mape']:.2f}%")
    print(f"  Samples: {metrics['n_samples']:,}")
    
    print("\n✅ THRESHOLD CHECKS:")
    print("-" * 40)
    
    all_passed = True
    for check_name, check_result in checks.items():
        status = "✅ PASS" if check_result["passed"] else "❌ FAIL"
        print(f"  {check_name}: {status}")
        if not check_result["passed"]:
            all_passed = False
    
    # Compile results
    results = {
        "validation_date": datetime.now().isoformat(),
        "passed": all_passed,
        "metrics": metrics,
        "thresholds": {
            "rmse": rmse_threshold,
            "r2": r2_threshold,
            "mae": mae_threshold
        },
        "checks": checks
    }
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 MODEL VALIDATION PASSED")
    else:
        print("⚠️  MODEL VALIDATION FAILED")
    print("="*60)
    
    return all_passed, results


def save_validation_report(results: dict, output_path: Path):
    """Save validation results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        return obj
    
    results_native = convert_to_native(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_native, f, indent=2)
    
    print(f"\n📄 Validation report saved to: {output_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate trained model against quality thresholds"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help=f"Path to models directory (default: {DEFAULT_MODEL_DIR})"
    )
    parser.add_argument(
        "--rmse-threshold",
        type=float,
        default=DEFAULT_RMSE_THRESHOLD,
        help=f"Maximum acceptable RMSE (default: {DEFAULT_RMSE_THRESHOLD})"
    )
    parser.add_argument(
        "--r2-threshold",
        type=float,
        default=DEFAULT_R2_THRESHOLD,
        help=f"Minimum acceptable R² (default: {DEFAULT_R2_THRESHOLD})"
    )
    parser.add_argument(
        "--mae-threshold",
        type=float,
        default=DEFAULT_MAE_THRESHOLD,
        help=f"Maximum acceptable MAE (default: {DEFAULT_MAE_THRESHOLD})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(VALIDATION_REPORT_PATH),
        help=f"Path to save validation report (default: {VALIDATION_REPORT_PATH})"
    )
    
    args = parser.parse_args()
    
    try:
        # Load model artifacts
        model_dir = Path(args.model_path)
        model, transformer, metadata = load_model_artifacts(model_dir)
        
        print(f"\n📦 Model Version: {metadata.get('version', 'unknown')}")
        print(f"   Model Type: {metadata.get('model_type', 'unknown')}")
        print(f"   Training Date: {metadata.get('training_date', 'unknown')}")
        
        # Load validation data
        data = load_validation_data()
        X, y = prepare_features(data)
        
        # Run validation
        passed, results = validate_model(
            model=model,
            transformer=transformer,
            X=X,
            y=y,
            rmse_threshold=args.rmse_threshold,
            r2_threshold=args.r2_threshold,
            mae_threshold=args.mae_threshold
        )
        
        # Add model metadata to results
        results["model_version"] = metadata.get("version", "unknown")
        results["model_type"] = metadata.get("model_type", "unknown")
        
        # Save report
        save_validation_report(results, Path(args.output))
        
        # Exit with appropriate code
        sys.exit(0 if passed else 1)
        
    except Exception as e:
        print(f"\n❌ VALIDATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
