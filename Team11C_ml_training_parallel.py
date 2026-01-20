"""
Team11C_ml_training_parallel.py
------------------------
PARALLEL Machine Learning Training Module - Supports Single Model Training

This module extends ml_training.py to support training individual models in parallel.
Key differences from original ml_training.py:
- train_single_model(): Trains ONE specific model (LinearRegression, RandomForest, or GBTRegressor)
- select_best_model(): Compares results from parallel tasks and registers the best one
- Designed for Airflow DAG parallelization

Usage:
    # Train single model
    python Team11C_ml_training_parallel.py --model LinearRegression
    
    # Train all models sequentially (original behavior)
    python Team11C_ml_training_parallel.py
"""

import os
import sys
import argparse

# Import all functions from original ml_training
from Team11C_ml_training import (
    init_spark, prepare_data, plot_metrics_comparison, 
    plot_actual_vs_predicted, plot_feature_importance,
    MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_REGISTRY_NAME,
    GOLD_PATH, REPORT_DIR, RMSE_THRESHOLD, R2_THRESHOLD
)

import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.spark
from pyspark.sql.functions import col, sqrt, avg, pow as spark_pow
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator


def get_model_config(model_name, stages_common, output_features):
    """
    Returns the configuration for a specific model.
    
    Args:
        model_name: 'LinearRegression', 'RandomForest', or 'GBTRegressor'
        stages_common: List of common pipeline stages (StringIndexer, VectorAssembler)
        output_features: List of feature names after preprocessing
    
    Returns:
        dict with keys: 'name', 'pipeline', 'grid'
    """
    if model_name == "LinearRegression":
        scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
        lr = LinearRegression(featuresCol="features", labelCol="price")
        pipeline_lr = Pipeline(stages=stages_common + [scaler, lr])
        grid_lr = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).build()
        return {"name": "LinearRegression", "pipeline": pipeline_lr, "grid": grid_lr}
    
    elif model_name == "RandomForest":
        rf = RandomForestRegressor(featuresCol="features_raw", labelCol="price", seed=42, maxBins=60)
        pipeline_rf = Pipeline(stages=stages_common + [rf])
        grid_rf = ParamGridBuilder() \
            .addGrid(rf.numTrees, [20, 50]) \
            .addGrid(rf.maxDepth, [5, 10]) \
            .build()
        return {"name": "RandomForest", "pipeline": pipeline_rf, "grid": grid_rf}
    
    elif model_name == "GBTRegressor":
        gbt = GBTRegressor(featuresCol="features_raw", labelCol="price", seed=42, maxBins=60)
        pipeline_gbt = Pipeline(stages=stages_common + [gbt])
        grid_gbt = ParamGridBuilder() \
            .addGrid(gbt.maxIter, [20, 50]) \
            .addGrid(gbt.maxDepth, [3, 5]) \
            .build()
        return {"name": "GBTRegressor", "pipeline": pipeline_gbt, "grid": grid_gbt}
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Use LinearRegression, RandomForest, or GBTRegressor")


def train_single_model(model_name):
    """
    Trains a SINGLE model and returns its metrics.
    
    Designed for parallel execution in Airflow DAG:
    - Task 1: train_single_model('LinearRegression')
    - Task 2: train_single_model('RandomForest')
    - Task 3: train_single_model('GBTRegressor')
    
    Args:
        model_name: 'LinearRegression', 'RandomForest', or 'GBTRegressor'
    
    Returns:
        dict: {
            'model': model_name,
            'rmse': float,
            'r2': float,
            'run_id': mlflow_run_id,
            'pred_min': float,
            'pred_max': float,
            'pred_mean': float
        }
    """
    # === INITIALIZATION ===
    spark = init_spark()
    spark.sparkContext.setLogLevel("ERROR")
    
    # === DATA PREPARATION ===
    df, num_cols, cat_cols = prepare_data(spark)
    
    # Cache loaded data since we'll use it for splitting and stats
    df.cache()
    total_records = df.count()  # Materialize cache
    
    # Split data
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    # Cache splits since they're reused multiple times in CV
    train_df.cache()
    test_df.cache()
    
    # Materialize caches
    train_count = train_df.count()
    test_count = test_df.count()
    
    print(f"\n[{model_name}] Dataset Statistics:")
    print(f"  Total records: {total_records}")
    print(f"  Training set: {train_count} records")
    print(f"  Test set: {test_count} records")
    
    # Unpersist original df since we only need splits
    df.unpersist()
    
    # === MLFLOW SETUP ===
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # === PIPELINE CONSTRUCTION ===
    stages_common = []
    output_features = []
    
    for c in cat_cols:
        stages_common.append(StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep"))
        output_features.append(f"{c}_idx")
    output_features += num_cols
    
    assembler = VectorAssembler(inputCols=output_features, outputCol="features_raw", handleInvalid="skip")
    stages_common.append(assembler)
    
    # Get model configuration
    conf = get_model_config(model_name, stages_common, output_features)
    
    # === TRAINING ===
    with mlflow.start_run(run_name=f"Parallel_{conf['name']}"):
        print(f"\n[{conf['name']}] Starting Grid Search with 3-Fold Cross-Validation...")
        
        evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")
        cv = CrossValidator(
            estimator=conf["pipeline"], 
            estimatorParamMaps=conf["grid"], 
            evaluator=evaluator, 
            numFolds=3, 
            seed=42
        )
        
        cv_model = cv.fit(train_df)
        predictions = cv_model.transform(test_df)
        
        # Calculate metrics
        manual_rmse = predictions.select(sqrt(avg(spark_pow(col("price") - col("prediction"), 2)))).collect()[0][0]
        rmse = evaluator.evaluate(predictions)
        r2 = evaluator.setMetricName("r2").evaluate(predictions)
        
        # Use manual RMSE if evaluator returns suspicious value
        if rmse < 10 and manual_rmse > 1000:
            print(f"  [WARNING] Using manual RMSE: {manual_rmse:.2f}")
            rmse = manual_rmse
        
        # Prediction statistics
        pred_stats = predictions.select("prediction").summary("min", "max", "mean").collect()
        pred_min = float(pred_stats[0][1])
        pred_max = float(pred_stats[1][1])
        pred_mean = float(pred_stats[2][1])
        
        print(f"\n[{conf['name']}] Results:")
        print(f"  RMSE: {rmse:.2f} €")
        print(f"  R²: {r2:.4f}")
        print(f"  Predictions range: [{pred_min:.2f}, {pred_max:.2f}], Mean: {pred_mean:.2f}")
        
        # === LOG TO MLFLOW ===
        mlflow.log_metrics({"rmse": rmse, "r2": r2})
        
        # Log best parameters
        best_stage = cv_model.bestModel.stages[-1]
        params = {"model_type": conf["name"], "cv_folds": 3}
        
        if hasattr(best_stage, "getRegParam"):
            params["regParam"] = best_stage.getRegParam()
        if hasattr(best_stage, "getNumTrees"):
            params["numTrees"] = best_stage.getNumTrees if not callable(best_stage.getNumTrees) else best_stage.getNumTrees()
        if hasattr(best_stage, "getMaxIter"):
            params["maxIter"] = best_stage.getMaxIter()
        if hasattr(best_stage, "getMaxDepth"):
            params["maxDepth"] = best_stage.getMaxDepth if not callable(best_stage.getMaxDepth) else best_stage.getMaxDepth()
        if hasattr(best_stage, "getMaxBins"):
            params["maxBins"] = best_stage.getMaxBins if not callable(best_stage.getMaxBins) else best_stage.getMaxBins()
        
        mlflow.log_params(params)
        
        # Log model
        mlflow.spark.log_model(cv_model.bestModel, "model")
        
        # Tags
        mlflow.set_tags({
            "model_family": conf["name"],
            "data_source": GOLD_PATH,
            "training_framework": "PySpark MLlib",
            "validation_strategy": "3-Fold CV + Hold-out Test",
            "project_phase": "B.1-Training-Parallel",
            "data_version": "v1.0"
        })
        
        run_id = mlflow.active_run().info.run_id
        
        # Generate individual model report (optional)
        report_subdir = os.path.join(REPORT_DIR, conf["name"])
        os.makedirs(report_subdir, exist_ok=True)
        
        best_preds_pd = predictions.select("price", "prediction").sample(False, 0.5, seed=42).toPandas()
        
        # Actual vs Predicted plot
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x="price", y="prediction", data=best_preds_pd, alpha=0.6)
        min_val = min(best_preds_pd["price"].min(), best_preds_pd["prediction"].min())
        max_val = max(best_preds_pd["price"].max(), best_preds_pd["prediction"].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Prediction")
        plt.title(f"Actual vs Predicted: {conf['name']}")
        plt.xlabel("Actual Price (€)")
        plt.ylabel("Predicted Price (€)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(report_subdir, "actual_vs_predicted.png"))
        plt.close()
        
        # Feature importance
        plot_feature_importance(cv_model.bestModel, output_features)
        
        # Move feature importance to model-specific folder
        if os.path.exists(os.path.join(REPORT_DIR, "feature_importance.png")):
            shutil.move(
                os.path.join(REPORT_DIR, "feature_importance.png"),
                os.path.join(report_subdir, "feature_importance.png")
            )
        
        # Log artifacts
        mlflow.log_artifact(report_subdir)
        
        print(f"\n[{conf['name']}] Training complete. Run ID: {run_id}")
    
    # Clean up cached DataFrames
    train_df.unpersist()
    test_df.unpersist()
    spark.stop()
    
    return {
        'model': conf['name'],
        'rmse': rmse,
        'r2': r2,
        'run_id': run_id,
        'pred_min': pred_min,
        'pred_max': pred_max,
        'pred_mean': pred_mean
    }


def select_best_model(model_results):
    """
    Compares results from parallel training tasks and registers the best model.
    
    Called after all 3 parallel training tasks complete in Airflow DAG.
    
    Args:
        model_results: List of dicts from train_single_model() containing:
            [
                {'model': 'LinearRegression', 'rmse': 45000, 'r2': 0.75, 'run_id': '...'},
                {'model': 'RandomForest', 'rmse': 42000, 'r2': 0.78, 'run_id': '...'},
                {'model': 'GBTRegressor', 'rmse': 40000, 'r2': 0.80, 'run_id': '...'}
            ]
    
    Returns:
        dict: Best model metadata
    """
    # Find best model (lowest RMSE)
    best = min(model_results, key=lambda x: x['rmse'])
    
    print("\n" + "="*60)
    print("MODEL SELECTION RESULTS")
    print("="*60)
    for result in sorted(model_results, key=lambda x: x['rmse']):
        marker = "🏆 WINNER" if result == best else "  "
        print(f"{marker} {result['model']:20s} | RMSE: {result['rmse']:>10.2f} € | R²: {result['r2']:.4f}")
    
    print("\n" + "="*60)
    print(f"BEST MODEL: {best['model']}")
    print(f"  RMSE: {best['rmse']:.2f} €")
    print(f"  R²: {best['r2']:.4f}")
    print(f"  Run ID: {best['run_id']}")
    print("="*60)
    
    # Create comparison chart
    os.makedirs(REPORT_DIR, exist_ok=True)
    results_df = pd.DataFrame(model_results)
    plot_metrics_comparison(results_df)
    
    # Register best model if it meets thresholds
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    if best['rmse'] < RMSE_THRESHOLD:
        print(f"\n✓ Model meets quality thresholds (RMSE < {RMSE_THRESHOLD})")
        
        try:
            model_uri = f"runs:/{best['run_id']}/model"
            
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=MODEL_REGISTRY_NAME,
                tags={
                    "model_type": best['model'],
                    "rmse": str(round(best['rmse'], 2)),
                    "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "training_mode": "parallel"
                }
            )
            
            print(f"✓ Model registered: {MODEL_REGISTRY_NAME} (Version {model_version.version})")
            
            # Transition to Staging
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=MODEL_REGISTRY_NAME,
                version=model_version.version,
                stage="Staging",
                archive_existing_versions=False
            )
            print(f"✓ Model transitioned to 'Staging' stage")
            
            # Add description
            client.update_model_version(
                name=MODEL_REGISTRY_NAME,
                version=model_version.version,
                description=f"Parallel-trained {best['model']} model. "
                            f"RMSE: {best['rmse']:.2f}€. "
                            f"Selected from 3 models trained in parallel with 3-fold CV."
            )
            
            # Tag the winning run
            client.set_tag(best['run_id'], "best_model", "true")
            client.set_tag(best['run_id'], "deployment_candidate", "true")
            
        except Exception as e:
            print(f"✗ Model Registry error: {e}")
    else:
        print(f"✗ Model does NOT meet quality thresholds (RMSE: {best['rmse']:.2f} >= {RMSE_THRESHOLD})")
    
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models (parallel mode)")
    parser.add_argument(
        "--model",
        choices=["LinearRegression", "RandomForest", "GBTRegressor", "all"],
        default="all",
        help="Specify which model to train (default: all models sequentially)"
    )
    
    args = parser.parse_args()
    
    if args.model == "all":
        # Sequential training (original behavior)
        print("Training all models sequentially...")
        results = []
        for model_name in ["LinearRegression", "RandomForest", "GBTRegressor"]:
            result = train_single_model(model_name)
            results.append(result)
        
        # Select and register best
        select_best_model(results)
    else:
        # Train single model (for parallel DAG execution)
        print(f"Training single model: {args.model}")
        result = train_single_model(args.model)
        print(f"\nModel training complete. Result: {result}")
