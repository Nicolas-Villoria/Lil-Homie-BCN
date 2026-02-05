"""
bda_rental_price_pipeline_optimized.py
---------------------------------------
OPTIMIZED Apache Airflow DAG with Parallel Data Collection + Parallel ML Training

KEY POINTS:
✅ Data Collection: 3 parallel tasks (Idealista, Income, Density) → 3x faster
✅ ML Training: 3 parallel tasks (LinearRegression, RandomForest, GBT) → 2-3x faster
✅ Model Selection: Automated best model comparison and registration
✅ Fault Isolation: Failed tasks don't block independent parallel tasks
✅ Auto-Recovery: MongoDB indexes self-heal on errors


Architecture:
    Collection (Parallel) → Formatting → Exploitation → ML Training (Parallel) → Model Selection
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ==============================================================================
# DAG DEFAULT ARGUMENTS
# ==============================================================================

default_args = {
    'owner': 'bda_team',
    'depends_on_past': False,
    'email': ['bda-team@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

# ==============================================================================
# TASK FUNCTIONS - PARALLEL DATA COLLECTION
# ==============================================================================

def collect_idealista_data(**context):
    """Collect Idealista property data (runs in parallel with Income/Density)"""
    from data_collection import run_collectors
    
    logical_date = context['logical_date']
    print(f"[Parallel Collection 1/3] Collecting Idealista data - {logical_date}")
    
    run_collectors(selected_collectors=['idealista'], reset=False)
    
    return {'dataset': 'idealista', 'status': 'success', 'logical_date': str(logical_date)}


def collect_income_data(**context):
    """Collect Income data (runs in parallel with Idealista/Density)"""
    from data_collection import run_collectors
    
    logical_date = context['logical_date']
    print(f"[Parallel Collection 2/3] Collecting Income data - {logical_date}")
    
    run_collectors(selected_collectors=['income'], reset=False)
    
    return {'dataset': 'income', 'status': 'success', 'logical_date': str(logical_date)}


def collect_density_data(**context):
    """Collect Density data (runs in parallel with Idealista/Income)"""
    from data_collection import run_collectors
    
    logical_date = context['logical_date']
    print(f"[Parallel Collection 3/3] Collecting Density data - {logical_date}")
    
    run_collectors(selected_collectors=['density'], reset=False)
    
    return {'dataset': 'density', 'status': 'success', 'logical_date': str(logical_date)}


# ==============================================================================
# TASK FUNCTIONS - SEQUENTIAL PROCESSING
# ==============================================================================

def run_data_formatting(**context):
    """
    Data Formatting (Silver Layer) - Single task for all datasets.
    MongoDB auto-recovery built-in via create_mongodb_indexes(auto_recover=True).
    """
    from data_formatting import run_formatting
    
    ti = context['ti']
    
    # Pull metadata from parallel collection tasks
    idealista_meta = ti.xcom_pull(task_ids='collect_idealista')
    income_meta = ti.xcom_pull(task_ids='collect_income')
    density_meta = ti.xcom_pull(task_ids='collect_density')
    
    print(f"[Data Formatting] Starting Silver layer processing")
    print(f"  Idealista: {idealista_meta}")
    print(f"  Income: {income_meta}")
    print(f"  Density: {density_meta}")
    
    run_formatting(selected_tasks=['all'])
    
    return {
        'status': 'success',
        'layer': 'silver',
        'datasets_formatted': ['idealista', 'income', 'density']
    }


def run_exploitation_zone(**context):
    """Exploitation Zone (Gold Layer) - Joins all datasets"""
    from exploitation_zone import create_exploitation_zone
    
    ti = context['ti']
    formatting_metadata = ti.xcom_pull(task_ids='data_formatting')
    
    print(f"[Exploitation Zone] Creating Gold layer")
    print(f"  Previous task: {formatting_metadata}")
    
    create_exploitation_zone()
    
    return {'status': 'success', 'layer': 'gold'}


# ==============================================================================
# TASK FUNCTIONS - PARALLEL ML TRAINING
# ==============================================================================

def train_linear_regression(**context):
    """Train Linear Regression model (runs in parallel with RF/GBT)"""
    from ml_training_parallel import train_single_model
    
    print("[Parallel ML 1/3] Training Linear Regression...")
    result = train_single_model('LinearRegression')
    
    # Return result via XCom for model selection task
    return result


def train_random_forest(**context):
    """Train Random Forest model (runs in parallel with LR/GBT)"""
    from ml_training_parallel import train_single_model
    
    print("[Parallel ML 2/3] Training Random Forest...")
    result = train_single_model('RandomForest')
    
    return result


def train_gbt(**context):
    """Train Gradient-Boosted Trees model (runs in parallel with LR/RF)"""
    from ml_training_parallel import train_single_model
    
    print("[Parallel ML 3/3] Training GBT...")
    result = train_single_model('GBTRegressor')
    
    return result


def select_and_register_best_model(**context):
    """
    Compares all 3 models and registers the best one.
    Runs after all parallel training tasks complete.
    """
    from ml_training_parallel import select_best_model
    
    ti = context['ti']
    
    # Pull results from all 3 parallel training tasks
    lr_result = ti.xcom_pull(task_ids='train_linear_regression')
    rf_result = ti.xcom_pull(task_ids='train_random_forest')
    gbt_result = ti.xcom_pull(task_ids='train_gbt')
    
    print("\n[Model Selection] Comparing results from parallel training tasks...")
    print(f"  Linear Regression: {lr_result}")
    print(f"  Random Forest: {rf_result}")
    print(f"  GBT: {gbt_result}")
    
    # Select best model and register to MLflow
    model_results = [lr_result, rf_result, gbt_result]
    best = select_best_model(model_results)
    
    return {
        'status': 'success',
        'best_model': best['model'],
        'best_rmse': best['rmse'],
        'best_r2': best['r2'],
        'best_run_id': best['run_id']
    }


def send_success_notification(**context):
    """Final task: Pipeline success summary"""
    ti = context['ti']
    logical_date = context['logical_date']
    
    # Pull metadata from all tasks
    idealista = ti.xcom_pull(task_ids='collect_idealista')
    income = ti.xcom_pull(task_ids='collect_income')
    density = ti.xcom_pull(task_ids='collect_density')
    formatting = ti.xcom_pull(task_ids='data_formatting')
    gold = ti.xcom_pull(task_ids='exploitation_zone')
    best_model = ti.xcom_pull(task_ids='select_best_model')
    
    print("\n" + "="*70)
    print("✅ PIPELINE EXECUTION SUMMARY - OPTIMIZED VERSION")
    print("="*70)
    print(f"Logical Date: {logical_date}")
    print(f"Pipeline Status: SUCCESS")
    print("\n[Phase 1] Data Collection (Parallel):")
    print(f"  ├─ Idealista: {idealista}")
    print(f"  ├─ Income: {income}")
    print(f"  └─ Density: {density}")
    print("\n[Phase 2] Data Processing (Sequential):")
    print(f"  ├─ Formatting: {formatting}")
    print(f"  └─ Gold Layer: {gold}")
    print("\n[Phase 3] ML Training (Parallel) & Selection:")
    print(f"  └─ Best Model: {best_model}")
    print("="*70)
    
    return {'pipeline_status': 'SUCCESS'}


# ==============================================================================
# DAG DEFINITION - OPTIMIZED STRUCTURE
# ==============================================================================

with DAG(
    dag_id='bda_rental_price_pipeline_optimized',
    default_args=default_args,
    description='OPTIMIZED: Parallel collection + parallel ML training',
    schedule='0 2 * * *',
    start_date=datetime(2025, 12, 1),
    catchup=False,
    tags=['bda', 'data-engineering', 'ml-pipeline', 'optimized', 'parallel'],
) as dag:
    
    dag.doc_md = """
    ## 🚀 OPTIMIZED Barcelona Rental Price Pipeline
    
    ### Architecture:
    ```
    ┌─ Idealista Collection ─┐
    ├─ Income Collection ────┤  (PARALLEL - 3x faster)
    └─ Density Collection ───┘
              ↓
         Formatting           (MONOLITHIC - efficient)
              ↓
       Exploitation Zone      (Joins all datasets)
              ↓
    ┌─ Train LR ─────────────┐
    ├─ Train RF ─────────────┤  (PARALLEL - 2-3x faster)
    └─ Train GBT ────────────┘
              ↓
       Model Selection        (Pick best, register to MLflow)
              ↓
        Notification
    ```
    
    ### Key Features:
    - ✅ **Parallel Data Collection**: 3 datasets collected simultaneously
    - ✅ **Parallel ML Training**: 3 models trained simultaneously
    - ✅ **Auto-Recovery**: MongoDB indexes self-heal on errors
    - ✅ **Fault Isolation**: Independent tasks can fail/retry separately
    - ✅ **Smart Selection**: Best model auto-registered to MLflow
    
    ### When to Use:
    - ✅ Production environments (fastest execution)
    - ✅ Multi-core machines (4+ cores recommended)
    - ✅ Unreliable data sources (fault isolation)
    - ✅ Large datasets (parallelism dominates overhead)
    """
    
    # ==============================================================================
    # PHASE 1: PARALLEL DATA COLLECTION
    # ==============================================================================
    
    task_collect_idealista = PythonOperator(
        task_id='collect_idealista',
        python_callable=collect_idealista_data,
    )
    
    task_collect_income = PythonOperator(
        task_id='collect_income',
        python_callable=collect_income_data,
    )
    
    task_collect_density = PythonOperator(
        task_id='collect_density',
        python_callable=collect_density_data,
    )
    
    # Synchronization barrier: Wait for all collections to complete
    task_collection_complete = EmptyOperator(
        task_id='collection_complete',
    )
    
    # ==============================================================================
    # PHASE 2: SEQUENTIAL DATA PROCESSING
    # ==============================================================================
    
    task_data_formatting = PythonOperator(
        task_id='data_formatting',
        python_callable=run_data_formatting,
    )
    
    task_exploitation_zone = PythonOperator(
        task_id='exploitation_zone',
        python_callable=run_exploitation_zone,
    )
    
    # ==============================================================================
    # PHASE 3: PARALLEL ML TRAINING
    # ==============================================================================
    
    task_train_lr = PythonOperator(
        task_id='train_linear_regression',
        python_callable=train_linear_regression,
    )
    
    task_train_rf = PythonOperator(
        task_id='train_random_forest',
        python_callable=train_random_forest,
    )
    
    task_train_gbt = PythonOperator(
        task_id='train_gbt',
        python_callable=train_gbt,
    )
    
    # Synchronization barrier: Wait for all models to train
    task_training_complete = EmptyOperator(
        task_id='training_complete',
    )
    
    # ==============================================================================
    # PHASE 4: MODEL SELECTION & DEPLOYMENT
    # ==============================================================================
    
    task_select_best = PythonOperator(
        task_id='select_best_model',
        python_callable=select_and_register_best_model,
    )
    
    task_success_notification = PythonOperator(
        task_id='success_notification',
        python_callable=send_success_notification,
        trigger_rule='all_success',
    )
    
    # ==============================================================================
    # TASK DEPENDENCIES - OPTIMIZED WORKFLOW
    # ==============================================================================
    
    # PHASE 1: Parallel data collection
    [task_collect_idealista, task_collect_income, task_collect_density] >> task_collection_complete
    
    # PHASE 2: Sequential processing (needs all data)
    task_collection_complete >> task_data_formatting >> task_exploitation_zone
    
    # PHASE 3: Parallel ML training (all models can train simultaneously)
    task_exploitation_zone >> [task_train_lr, task_train_rf, task_train_gbt] >> task_training_complete
    
    # PHASE 4: Model selection and notification
    task_training_complete >> task_select_best >> task_success_notification
