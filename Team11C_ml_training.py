"""
Team11C_ml_training.py
-----------------------
Machine Learning Training Module (Gold (Exploitation Zone) -> ML Models).

This script implements an automated ML pipeline for predicting property rental prices in Barcelona.
It performs the following operations:
1. Reads curated data from the Gold Layer (Delta Lake).
2. Trains multiple regression models (Linear Regression, Random Forest, GBT) with hyperparameter tuning.
3. Uses MLflow for experiment tracking, model registry, and versioning.
4. Generates visual reports (metric comparisons, predictions, feature importance).
5. Registers the best model for production deployment.
"""

import os
import sys

# Environmental configurations for matplotlib and macOS fork safety
os.environ['MPLBACKEND'] = 'Agg'  # Non-interactive matplotlib backend
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'  # Disable macOS fork safety check

import shutil
import warnings
import numpy as np
import pandas as pd

# CRITICAL: Set matplotlib backend to non-GUI before importing pyplot
# This prevents macOS fork() issues with Airflow
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.utils.requirements_utils")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sqrt, avg, pow as spark_pow
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from delta import *

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================

# MLflow experiment name for tracking all training runs
MLFLOW_EXPERIMENT_NAME = "bda_project_price_prediction"

# MLflow tracking URI (set to local filesystem for this project)
MLFLOW_TRACKING_URI = "file:./mlruns"

# Model Registry name for production deployment
MODEL_REGISTRY_NAME = "BarcelonaRentalPriceModel"

# Path to the Gold Layer Delta table containing curated property data
GOLD_PATH = "data_lake/gold/property_prices"

# Output directory for generated visual reports and artifacts
REPORT_DIR = "reports"

# Model performance thresholds for production eligibility
RMSE_THRESHOLD = 150000  # Maximum acceptable RMSE (in euros, ~5% of mean price)
R2_THRESHOLD = 0.6       # Minimum acceptable R² score

def init_spark():
    """
    Initializes and configures a Spark Session with Delta Lake support.
    Suppresses verbose Spark/Ivy startup logs.
    """
    # Force Spark to bind to localhost
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
    
    # Ensure PySpark workers use the same Python environment
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    # Suppress stderr (fd 2) and stdout (fd 1) to hide Spark/Ivy startup logs
    stderr_fd = sys.stderr.fileno()
    stdout_fd = sys.stdout.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    saved_stdout_fd = os.dup(stdout_fd)

    with open(os.devnull, 'w') as devnull:
        os.dup2(devnull.fileno(), stderr_fd)
        os.dup2(devnull.fileno(), stdout_fd)
        try:
            builder = SparkSession.builder \
                .appName("ML_AutoML_GBT") \
                .master("local[*]") \
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            
            spark = configure_spark_with_delta_pip(builder).getOrCreate()
            
            # Set log level to ERROR to suppress warnings
            spark.sparkContext.setLogLevel("ERROR")
        finally:
            os.dup2(saved_stderr_fd, stderr_fd)
            os.dup2(saved_stdout_fd, stdout_fd)
            os.close(saved_stderr_fd)
            os.close(saved_stdout_fd)
    
    return spark

def prepare_data(spark):
    """
    Loads the Gold Layer dataset and identifies feature columns for ML training.

    Args:
        spark (SparkSession): Active Spark session for data operations
    
    Returns:
        A 3-element tuple containing:
            - df (DataFrame): Cleaned Spark DataFrame ready for training
            - numeric_features (list[str]): List of numeric column names
            - categorical_features (list[str]): List of categorical column names
    """
    # Read data from Gold Layer Delta table
    df = spark.read.format("delta").load(GOLD_PATH)
    
    # Define feature groups (based on Gold Layer schema)
    numeric_features = ["size", "rooms", "bathrooms", "avg_income_index", "density_val"]
    categorical_features = ["neighborhood", "propertyType", "district"]
    
    # Validation: Keep only features that actually exist in the loaded data
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]
    
    # Data Cleaning: Remove rows with missing values in features or target
    df = df.dropna(subset=numeric_features + ["price"])
    
    return df, numeric_features, categorical_features

# --- Visualization Functions ---

def plot_metrics_comparison(results_df):
    """
    Generates separate bar charts for RMSE and R² comparison.
    
    Creates two subplots with independent scales to avoid visualization issues
    when RMSE values (large numbers) and R² values (0-1 range) are very different.
    
    Args:
        results_df (pandas.DataFrame): DataFrame with columns ['model', 'rmse', 'r2']
    
    Output:
        Saves 'metric_comparison.png' with two subplots showing RMSE and R² separately
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: RMSE (Lower is better)
    sns.barplot(x="model", y="rmse", data=results_df, palette="Reds_r", ax=ax1, hue="model", legend=False)
    ax1.set_title("RMSE Comparison (Lower is Better)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("RMSE (€)", fontsize=10)
    ax1.set_xlabel("Model", fontsize=10)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.2f', padding=3)
    
    # Plot 2: R² (Higher is better)
    sns.barplot(x="model", y="r2", data=results_df, palette="Greens", ax=ax2, hue="model", legend=False)
    ax2.set_title("R² Comparison (Higher is Better)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("R² Score", fontsize=10)
    ax2.set_xlabel("Model", fontsize=10)
    ax2.set_ylim([0, 1])  # R² is always between 0 and 1
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.4f', padding=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "metric_comparison.png"), dpi=100, bbox_inches='tight')
    plt.close()


def plot_actual_vs_predicted(predictions_pd, model_name):
    """
    Generates a scatter plot of actual vs predicted prices.
    
    Args:
        predictions_pd (pandas.DataFrame): DataFrame with 'price' and 'prediction' columns
        model_name (str): Name of the model being visualized (for plot title)
    
    Output:
        Saves 'actual_vs_predicted.png' to REPORT_DIR
    
    Interpretation:
        - Red dashed line: Perfect prediction (y = x)
        - Points above line: Model under-predicts (predicted < actual)
        - Points below line: Model over-predicts (predicted > actual)
        - Tight clustering around diagonal: Good model fit
        - Wide spread: High prediction variance/error
    
    Note:
        This plot is essential for detecting:
        - Systematic bias (consistent over/under prediction)
        - Heteroscedasticity (error variance changing with price level)
        - Outliers (extreme prediction errors)
    """
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x="price", y="prediction", data=predictions_pd, alpha=0.6)
    
    # Add diagonal reference line representing perfect predictions (y = x)
    min_val = min(predictions_pd["price"].min(), predictions_pd["prediction"].min())
    max_val = max(predictions_pd["price"].max(), predictions_pd["prediction"].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Prediction")
    
    plt.title(f"Actual vs Predicted: {model_name}")
    plt.xlabel("Actual Price (€)")
    plt.ylabel("Predicted Price (€)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "actual_vs_predicted.png"))
    plt.close()

def plot_feature_importance(model, feature_names):
    """
    Extracts and visualizes feature importance scores.
    
    This provides a GLOBAL explanation of which features the model relies on most
    for making predictions. It helps answer: "What factors drive rental prices?"
    
    Args:
        model (PipelineModel): Trained Spark ML Pipeline (must contain a model with
                               featureImportances or coefficients attribute)
        feature_names (list[str]): Ordered list of feature names matching the model's
                                   feature vector indices
    
    Output:
        Saves 'feature_importance.png' showing top 15 most important features
    
    Importance Calculation:
        - Tree Models (RF, GBT): Uses Gini importance (measures average split gain)
        - Linear Models: Uses absolute coefficient values (feature weight magnitude)
    
    Interpretation:
        - Higher score = Feature has more influence on predictions
        - For pricing: High importance might indicate key property attributes
          (e.g., size, neighborhood) or economic indicators (e.g., income index)
    
    Note:
        - Feature importance shows correlation, not causation
        - Does NOT explain individual predictions (use SHAP for local explanations)
        - Categorical features appear as "<feature>_idx" after StringIndexer
    """
    # Extract the trained estimator (last stage in the pipeline)
    last_stage = model.stages[-1]
    scores = []
    
    # Importance extraction varies by model type
    if hasattr(last_stage, "featureImportances"): # Tree-based models (RF, GBT)
        scores = last_stage.featureImportances.toArray()
    elif hasattr(last_stage, "coefficients"): # Linear Regression
        scores = np.abs(last_stage.coefficients.toArray())  # Absolute value for magnitude
    else:
        # Model doesn't support feature importance (e.g., some custom estimators)
        return

    # Handle potential feature name/score length mismatch (safety check)
    if len(scores) != len(feature_names):
        feat_labels = [f"Feat_{i}" for i in range(len(scores))]
    else:
        feat_labels = feature_names

    # Create DataFrame and sort by importance (descending)
    df_imp = pd.DataFrame({"feature": feat_labels, "importance": scores})
    df_imp = df_imp.sort_values(by="importance", ascending=False).head(15)

    # Plot horizontal bar chart (easier to read feature names)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=df_imp, hue="feature", palette="magma", legend=False)
    plt.title("Global Feature Importance (Top 15)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "feature_importance.png"), dpi=100, bbox_inches='tight')
    plt.close()

# --- Main Training Flow ---

def train_regression_models():
    """
    Orchestrates the complete ML training workflow for property price prediction,
    implementing the full ML pipeline as specified in project requirements:
    
    Workflow Steps performed:
    We first initialize Spark session and prepare output directories. Then we 
    load Gold Layer (B.1 - Data) data and split into train/test (80/20) before training 
    the 3 models (B.1 - Training). Each model, which are Linear Regression, Random Forest and 
    Gradient-Boosted Trees, is trained respectively with 3-Fold Cross-Validation and 
    hyperparameter tuning via Grid Search. After we [B.2 - Tracking] log all experiments to MLflow 
    (params, metrics, models) and [B.3 - Reporting] generate visual analytics 
    (metrics, predictions, importance). Finally, we [B.2 - Deployment] register the best model 
    in MLflow Model Registry if it meets quality thresholds.
    
    Automation Features:
    This workflow includes automated hyperparameter tuning via Grid Search, 3-Fold 
    Cross-Validation for robust performance estimation, and a held-out test set (20%) 
    for final unbiased evaluation. The best model is automatically selected based on RMSE (lower is better).
    
    Model Configurations:
    Specified models have the following configurations: L
    - Linear Regression: Tests regularization strengths (0.1, 0.01)
    - Random Forest: Tests tree counts (20, 50) and depths (5, 10)
    - GBT: Tests iterations (20, 50) and depths (3, 5)
    
    Output Artifacts:
    - MLflow: Logged runs with metrics, parameters, and model binaries
    - Reports: PNG visualizations and CSV metrics in 'reports/' directory
    
    Returns:
        None (outputs are logged to MLflow and saved to disk)
    """
    # === INITIALIZATION ===
    spark = init_spark()
    spark.sparkContext.setLogLevel("ERROR")  # Suppress verbose Spark logs
    
    # Clean and recreate reports directory for fresh outputs
    if os.path.exists(REPORT_DIR): shutil.rmtree(REPORT_DIR)
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # === [TASK B.1] DATA PREPARATION ===
    df, num_cols, cat_cols = prepare_data(spark)
    
    # Cache the loaded DataFrame since it will be used for splitting and stats
    df.cache()
    
    # Log dataset statistics (triggers cache materialization)
    total_records = df.count()
    print(f"\nDataset Statistics:")
    print(f"  Total records: {total_records}")
    print(f"  Numeric features: {len(num_cols)}")
    print(f"  Categorical features: {len(cat_cols)}")
    
    # Split data: 80% training (used for CV), 20% held-out test set
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    # Cache train/test splits since they're reused across all 3 models
    train_df.cache()
    test_df.cache()
    
    # Materialize caches with counts
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"  Training set: {train_count} records ({train_count/total_records*100:.1f}%)")
    print(f"  Test set: {test_count} records ({test_count/total_records*100:.1f}%)")
    
    # Unpersist original df since we only need the splits now
    df.unpersist()
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # === [TASK B.1] PIPELINE CONSTRUCTION ===
    # Build common preprocessing stages (shared by all models)
    stages_common = []
    output_features = []
    
    # Step 1: Encode categorical features to numeric indices
    # (StringIndexer assigns unique integer IDs to categories)
    for c in cat_cols:
        stages_common.append(StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep"))
        output_features.append(f"{c}_idx")
    output_features += num_cols
    
    # Step 2: Combine all features into a single vector column
    # (Required by Spark ML algorithms which expect a 'features' column)
    assembler = VectorAssembler(inputCols=output_features, outputCol="features_raw", handleInvalid="skip")
    stages_common.append(assembler)
    
    models_conf = []

    # Model A: Linear Regression (Baseline)
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
    lr = LinearRegression(featuresCol="features", labelCol="price")
    pipeline_lr = Pipeline(stages=stages_common + [scaler, lr])
    grid_lr = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).build()
    models_conf.append({"name": "LinearRegression", "pipeline": pipeline_lr, "grid": grid_lr})

    # Model B: Random Forest (Bagging)
    # maxBins must be >= max number of categories in any categorical feature (neighborhood has 51 values)
    # seed=42 ensures reproducibility across runs
    rf = RandomForestRegressor(featuresCol="features_raw", labelCol="price", seed=42, maxBins=60)
    pipeline_rf = Pipeline(stages=stages_common + [rf])
    grid_rf = ParamGridBuilder() \
        .addGrid(rf.numTrees, [20, 50]) \
        .addGrid(rf.maxDepth, [5, 10]) \
        .build()
    models_conf.append({"name": "RandomForest", "pipeline": pipeline_rf, "grid": grid_rf})

    # Model C: Gradient-Boosted Trees (Boosting - Spark Native)
    # GBTs train sequentially, so we keep tree count lower for speed
    # maxBins must be >= max number of categories in any categorical feature (neighborhood has 51 values)
    # seed=42 ensures reproducibility across runs
    gbt = GBTRegressor(featuresCol="features_raw", labelCol="price", seed=42, maxBins=60)
    pipeline_gbt = Pipeline(stages=stages_common + [gbt])
    grid_gbt = ParamGridBuilder() \
        .addGrid(gbt.maxIter, [20, 50]) \
        .addGrid(gbt.maxDepth, [3, 5]) \
        .build()
    models_conf.append({"name": "GBTRegressor", "pipeline": pipeline_gbt, "grid": grid_gbt})

    best_run_info = {"rmse": float("inf"), "run_id": None, "name": None, "predictions": None, "model": None}
    results = []
    evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")

    # === [TASK B.1] TRAINING LOOP (Grid Search + Cross-Validation) ===
    for conf in models_conf:
        with mlflow.start_run(run_name=f"GridSearch_{conf['name']}"):
            print(f"Running Grid Search for {conf['name']}...")
            
            # Cross Validation: Trains model with each parameter combo
            # using 3-fold CV to estimate generalization performance
            cv = CrossValidator(estimator=conf["pipeline"], estimatorParamMaps=conf["grid"], evaluator=evaluator, numFolds=3, seed=42)
            cv_model = cv.fit(train_df)  # Returns best model from grid
            
            # Evaluate the BEST model from CV on held-out test set
            # (This gives unbiased performance estimate)
            predictions = cv_model.transform(test_df)
            
            # Manual RMSE calculation to verify evaluator (Spark bug workaround)
            manual_rmse = predictions.select(sqrt(avg(spark_pow(col("price") - col("prediction"), 2)))).collect()[0][0]
            
            # Use Spark's evaluator
            rmse = evaluator.evaluate(predictions)
            r2 = evaluator.setMetricName("r2").evaluate(predictions)
            
            # Use manual RMSE if evaluator returns suspicious value (< 10)
            if rmse < 10 and manual_rmse > 1000:
                print(f"  [WARNING] Evaluator returned suspicious RMSE: {rmse:.2f}, using manual calculation: {manual_rmse:.2f}")
                rmse = manual_rmse
            
            # Validation: Check prediction range to detect anomalies
            pred_stats = predictions.select("prediction").summary("min", "max", "mean").collect()
            pred_min = float(pred_stats[0][1])
            pred_max = float(pred_stats[1][1])
            pred_mean = float(pred_stats[2][1])
            
            print(f"  Result -> RMSE: {rmse:.2f} | R²: {r2:.4f}")
            print(f"  Predictions range: [{pred_min:.2f}, {pred_max:.2f}], Mean: {pred_mean:.2f}")
            
            # === [TASK B.2] LOG TO MLFLOW ===
            # Log metrics
            mlflow.log_metrics({"rmse": rmse, "r2": r2})
            
            # Log Best Parameters
            best_stage = cv_model.bestModel.stages[-1]
            params = {"model_type": conf["name"], "cv_folds": 3}
            
            # Handle both method calls and property access
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
            
            # Log model with metadata
            mlflow.spark.log_model(
                cv_model.bestModel, 
                "model",
                registered_model_name=None  # Will register best model separately
            )
            
            # Add comprehensive tags for model governance
            mlflow.set_tags({
                "model_family": conf["name"],
                "data_source": GOLD_PATH,
                "training_framework": "PySpark MLlib",
                "validation_strategy": "3-Fold CV + Hold-out Test",
                "project_phase": "B.1-Training",
                "data_version": "v1.0"
            })

            results.append({"model": conf["name"], "rmse": rmse, "r2": r2})
            
            if rmse < best_run_info["rmse"]:
                best_run_info = {
                    "rmse": rmse, 
                    "run_id": mlflow.active_run().info.run_id, 
                    "name": conf["name"],
                    "model": cv_model.bestModel,
                    "predictions": predictions
                }

    # === [TASK B.3] GENERATE VISUAL REPORTS ===
    print("\nGenerating Visualizations...")
    
    # A. Metric Comparison Bar Chart (Compare all models)
    plot_metrics_comparison(pd.DataFrame(results))
    
    # B. Actual vs Predicted Scatter Plot (Best Model only)
    # Sample 50% of predictions to reduce plot density while maintaining pattern visibility
    best_preds_pd = best_run_info["predictions"].select("price", "prediction").sample(False, 0.5, seed=42).toPandas()
    plot_actual_vs_predicted(best_preds_pd, best_run_info["name"])
    
    # C. Feature Importance Chart (Global Interpretability)
    plot_feature_importance(best_run_info["model"], output_features)

    # === [TASK B.2] MODEL REGISTRY & DEPLOYMENT ===
    if best_run_info["run_id"]:
        print(f"\nLogging reports to Best Run ({best_run_info['name']})...")
        with mlflow.start_run(run_id=best_run_info["run_id"]):
            # Attach all visual reports to the best model's MLflow run
            mlflow.log_artifact(REPORT_DIR)
            
            # Tag for easy filtering in MLflow UI
            mlflow.set_tags({
                "deployment_candidate": "true",
                "best_model": "true",
                "champion_model": best_run_info["name"]
            })
        
        # Register the best model in MLflow Model Registry
        print(f"\n{'='*60}")
        print("MODEL REGISTRY: Registering Best Model")
        print(f"{'='*60}")
        
        model_uri = f"runs:/{best_run_info['run_id']}/model"
        
        # Check if model meets production thresholds
        meets_threshold = (best_run_info["rmse"] < RMSE_THRESHOLD)
        
        if meets_threshold:
            print(f"✓ Model meets quality thresholds (RMSE < {RMSE_THRESHOLD})")
            
            try:
                # Register model
                model_version = mlflow.register_model(
                    model_uri=model_uri,
                    name=MODEL_REGISTRY_NAME,
                    tags={
                        "model_type": best_run_info["name"],
                        "rmse": str(round(best_run_info["rmse"], 2)),
                        "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                )
                
                print(f"✓ Model registered: {MODEL_REGISTRY_NAME} (Version {model_version.version})")
                
                # Transition to Staging for validation
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=MODEL_REGISTRY_NAME,
                    version=model_version.version,
                    stage="Staging",
                    archive_existing_versions=False
                )
                print(f"✓ Model transitioned to 'Staging' stage for validation")
                
                # Add model version description
                client.update_model_version(
                    name=MODEL_REGISTRY_NAME,
                    version=model_version.version,
                    description=f"Grid Search trained {best_run_info['name']} model. "
                                f"RMSE: {best_run_info['rmse']:.2f}€. "
                                f"Trained with 3-fold CV and hyperparameter tuning on {train_count} samples."
                )
                
                print(f"\n{'='*60}")
                print("DEPLOYMENT INSTRUCTIONS:")
                print(f"{'='*60}")
                print(f"1. Review model in MLflow UI: http://localhost:5000 or http://<your-ip>:5000")
                print(f"2. Validate model in Staging environment")
                print(f"3. Transition to Production: ")
                print(f"   mlflow models transition --name {MODEL_REGISTRY_NAME} ")
                print(f"   --version {model_version.version} --stage Production")
                print(f"4. Serve model: mlflow models serve -m 'models:/{MODEL_REGISTRY_NAME}/Production'")
                
            except Exception as e:
                print(f"✗ Model Registry error: {e}")
                print("Note: Ensure MLflow tracking server is running for Model Registry features.")
        else:
            print(f"✗ Model does NOT meet quality thresholds (RMSE: {best_run_info['rmse']:.2f} >= {RMSE_THRESHOLD})")
            print("Model will NOT be registered for production deployment.")

    # === FINAL SUMMARY ===
    print("\n" + "="*40)
    print(f"WINNER: {best_run_info['name']} (RMSE: {best_run_info['rmse']:.2f})")
    print("="*40)
    print(f"\nAll results logged to MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
    print(f"Visual reports saved to: {REPORT_DIR}/")
    print(f"{'='*60}")
    print("VIEW RESULTS IN MLFLOW UI:")
    print(f"{'='*60}")
    print("1. Start MLflow UI server (accessible from network):")
    print("   # With virtual environment:")
    print("   venv/bin/mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5001")
    print("\n   # Without virtual environment:")
    print("   mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5001")
    print("\n2. Open browser to:")
    print("   - Local:  http://localhost:5001")
    print("   - Remote: http://<your-ip>:5001 (replace <your-ip> with actual IP)")
    print("\n3. Navigate to:")
    print(f"   - Experiments → '{MLFLOW_EXPERIMENT_NAME}' (compare all runs)")
    print(f"   - Models → '{MODEL_REGISTRY_NAME}' (view registered model)")
    
    # Clean up cached DataFrames and Spark resources
    train_df.unpersist()
    test_df.unpersist()
    spark.stop()

if __name__ == "__main__":
    train_regression_models()
