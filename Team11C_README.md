# Barcelona Rental Price Estimation

A Big Data Analytics pipeline for predicting property prices in Barcelona using real estate listings and neighborhood socioeconomic data.

## Table of Contents

- [About the Project](#about-the-project)
- [Authors](#authors)
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Testing](#testing)
- [Design Decisions](#design-decisions)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## About the Project

This project builds an end-to-end data pipeline that predicts rental and sale prices of properties in Barcelona. We collect data from multiple sources, clean and transform it through a medallion architecture (Bronze, Silver, Gold layers), and train machine learning models to predict prices based on property characteristics and neighborhood features.

The pipeline handles:
- Data ingestion from raw JSON and CSV files into a Bronze layer (Parquet format)
- Data cleaning and standardization into a Silver layer (MongoDB)
- Feature engineering and joins into a Gold layer (Delta Lake)
- Training of regression models (Linear Regression, Random Forest, Gradient Boosted Trees) with hyperparameter tuning
- Experiment tracking and model registry with MLflow
- Workflow orchestration with Apache Airflow

Data sources used:
- Idealista: Real estate listings with property attributes (size, rooms, location, price)
- Population Density: Neighborhood demographic statistics from Barcelona Open Data
- Income Data: Average income levels per neighborhood from Barcelona Open Data

---

## Authors

Team 11C - BDA Course

- Nicolás Villoria Alonso
- Pablo Fernández Aulet

---

## Architecture

We implemented a Medallion Architecture with three data layers:

```
Raw Data (datasets/)
        |
        v
+------------------+
|   BRONZE LAYER   |  Raw data ingestion (Parquet)
|   Landing Zone   |  Script: Team11C_data_collection.py
+------------------+
        |
        v
+------------------+
|   SILVER LAYER   |  Cleaned and standardized data (MongoDB)
|   Cleaned Zone   |  Script: Team11C_data_formatting.py
+------------------+
        |
        v
+------------------+
|    GOLD LAYER    |  ML-ready features (Delta Lake)
| Exploitation Zone|  Script: Team11C_exploitation_zone.py
+------------------+
        |
        v
+------------------+
|   ML PIPELINE    |  Model training and registry (MLflow)
|                  |  Script: Team11C_ml_training.py
+------------------+
```

The Airflow DAG orchestrates the full pipeline:

```
Data Collection (Bronze)
        |
        v
Data Formatting (Silver)
        |
        v
Exploitation Zone (Gold)
        |
        v
ML Training (Models)
```

We also have an optimized DAG (`Team11C_bda_rental_price_pipeline_optimized`) that runs data collectors in parallel and trains the three ML models in parallel, which reduces total execution time by about 50%.

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Apache Spark (PySpark) | Distributed data processing |
| Delta Lake | ACID transactions for the Gold layer |
| MongoDB | Document storage for the Silver layer |
| Apache Airflow | Workflow orchestration |
| MLflow | Experiment tracking and model registry |
| Python 3.9+ | Main programming language |
| Java 8/11 | Required by Spark |

---

## Prerequisites

### ⚠️ Important for Windows Users

**Windows users MUST use WSL2 (Windows Subsystem for Linux) to run this project.** Apache Airflow does not support native Windows, and you will encounter compatibility issues with signals, file locking, and other POSIX-dependent features.

**To install WSL2:**
```powershell
# Run in PowerShell as Administrator
wsl --install
# Restart your computer when prompted
# After restart, open "Ubuntu" from Start menu and follow Linux instructions below
```

Once WSL2 is installed, access your Windows files at `/mnt/c/` and follow the Linux installation steps.

---

### System Requirements (All Platforms)

Before installing, make sure you have:

1. **Python 3.9 or higher**
2. **Java 8 or 11** (required by PySpark)
3. **MongoDB 5.0 or higher**
4. **WSL2** (Windows users only - see above)

To check your installations:

```bash
python3 --version
java -version
mongod --version
```

MongoDB must be running before executing the pipeline:

```bash
# Start MongoDB (Linux / WSL2)
sudo systemctl start mongod

# Start MongoDB (macOS with Homebrew)
brew services start mongodb-community
```

---

## Installation

### For Windows Users (WSL2)

**⚠️ Windows users: Follow these steps inside WSL2 (Ubuntu terminal), not in PowerShell or CMD.**

1. Open WSL2 (Ubuntu) from the Start menu

2. Navigate to your project location:
```bash
# If the project is on your C: drive
cd /mnt/c/Users/YourUsername/path/to/project
# Or clone it directly in WSL2 home directory
cd ~
```

3. Clone the repository (if not already present):
```bash
git clone https://github.com/FernanESP0/Proyecto_BDA_2.git
cd Proyecto_BDA_2
```

4. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

5. Continue with step 3 below (Install dependencies)

### For macOS/Linux Users

1. Clone the repository. If the project folder already exists on your machine, skip this and go directly to Step 2 inside that directory.

```bash
git clone https://github.com/FernanESP0/Proyecto_BDA_2.git
cd Proyecto_BDA_2
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

### All Platforms (Continue Here)

3. Install dependencies. This process may take a few minutes depending on your internet connection and system hardware. Estimated time: ~15 minutes.

```bash
pip install --upgrade pip
pip install -r Team11C_requirements.txt
```

4. Verify the installation:

```bash
python3 -c "import pyspark; print('PySpark OK')"
python3 -c "import mlflow; print('MLflow OK')"
python3 -c "import pymongo; pymongo.MongoClient('mongodb://localhost:27017/'); print('MongoDB OK')"
```

5. Initialize Airflow (one-time setup):

**Note for Windows users:** Make sure you're running this inside WSL2, not in PowerShell.

```bash
python3 Team11C_airflow_setup.py
```

---

## Running the Project

There are two ways to run the pipeline:

### Option 1: Using Airflow (Recommended)

This runs the full pipeline with orchestration and monitoring.

1. Start Airflow:

**For WSL2 users:** First convert the script to Unix line endings:
```bash
# Install dos2unix (one-time only)
sudo apt-get update && sudo apt-get install -y dos2unix

# Convert the script
dos2unix Team11C_start_airflow.sh

# Make it executable and run
chmod +x Team11C_start_airflow.sh
./Team11C_start_airflow.sh
```

**For macOS/Linux users:**
```bash
chmod +x Team11C_start_airflow.sh
./Team11C_start_airflow.sh
```

**2. Access the Dashboard** Open the Airflow UI in your browser at `http://localhost:8080`.

**3. Authentication** Log in with the credentials provided in your terminal during startup (typically found in `airflow/standalone_admin_password.txt`).

**4. Locate the Pipeline** Search for the DAG titled: `Team11C_bda_rental_price_pipeline_optimized`.

**5. Initialize the Run** Toggle the DAG to **"On"** using the switch on the left, then click the **Play (Trigger DAG)** button on the right to start the execution.

**6. Monitor & Manage Execution** Track progress through the **Grid** or **Graph View**.

* **Cold Start Factor:** Be aware that the first run may experience a "cold start" delay (3–5 minutes) as the Spark session initializes and connections to MongoDB and MLflow are established.
* **Error Handling:** If a task fails due to transient connection issues or resource timeouts, simply click the failed (red) task instance and select the **"Clear"** button to restart it.


### Option 2: Running Scripts Manually

Run each step of the pipeline individually:

```bash
# Step 1: Data Collection (Bronze Layer)
python3 Team11C_data_collection.py --collectors all

# Step 2: Data Formatting (Silver Layer)
python3 Team11C_data_formatting.py --tasks all

# Step 3: Feature Engineering (Gold Layer)
python3 Team11C_exploitation_zone.py

# Step 4: ML Training
python3 Team11C_ml_training.py
```

To reset and reload all data from scratch:

```bash
python3 Team11C_data_collection.py --collectors all --reset
```

### Viewing Results

After the pipeline completes:

- Check the reports in `reports/` (metric comparisons, prediction plots, feature importance)
- Start the MLflow UI to see experiment tracking:

```bash
mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5000
```

Then open http://localhost:5000

---

## Testing

Testing was done manually by running the pipeline and checking the outputs at each stage:

1. Bronze Layer: Check that Parquet files are created in `data_lake/bronze/` and contain the expected records

2. Silver Layer: Verify MongoDB collections have the cleaned data:
   ```bash
   mongosh bda_project --eval "db.idealista.countDocuments({})"
   mongosh bda_project --eval "db.income.countDocuments({})"
   mongosh bda_project --eval "db.density.countDocuments({})"
   ```

3. Gold Layer: Confirm Delta table exists and has joined records:
   ```bash
   python3 -c "
   from pyspark.sql import SparkSession
   from delta import configure_spark_with_delta_pip
   builder = SparkSession.builder.appName('Test')
   spark = configure_spark_with_delta_pip(builder).getOrCreate()
   df = spark.read.format('delta').load('data_lake/gold/property_prices')
   print(f'Gold layer records: {df.count()}')
   "
   ```

4. ML Models: Check that models are logged in MLflow and reports are generated in `reports/`

5. Airflow: Verify all tasks show green (success) in the Airflow UI

---

## Design Decisions

### Why a Medallion Architecture?

We chose the Bronze-Silver-Gold pattern because it provides clear separation of concerns:
- Bronze keeps raw data intact for reproducibility
- Silver handles data quality without modifying the source
- Gold contains only what the ML models need

This also makes debugging easier since we can inspect data at each stage.

### Why MongoDB for the Silver Layer?

MongoDB works well for our semi-structured data. The property listings have variable schemas (some properties have parking, others don't), and MongoDB handles this flexibility without requiring schema migrations. The upsert strategy also makes the pipeline idempotent.

### Why Delta Lake for the Gold Layer?

Delta Lake gives us ACID transactions, which prevents partial writes if the job fails midway. It also supports time travel, so we can query previous versions of the data if needed.

### Why Three ML Models?

We train Linear Regression (as a baseline), Random Forest, and Gradient Boosted Trees because they have different strengths:
- Linear Regression is interpretable and fast
- Random Forest handles non-linear relationships and is robust to outliers
- GBT often achieves the best accuracy for tabular data

We use grid search with cross-validation to tune hyperparameters and automatically register the best model.

### Why Parallel Execution in the Optimized DAG?

The data collectors are independent (Idealista, Income, Density don't depend on each other), so running them in parallel saves time. Same for the three ML models. The optimized DAG cuts total execution time roughly in half.

### Incremental Loading

The data collection script tracks which files have been processed in log files. This means we only process new files on subsequent runs instead of reloading everything.

### Broadcast Joins

The Income and Density tables are small (around 73 neighborhoods), so we broadcast them during joins with the larger Idealista table. This avoids shuffling the large table across the cluster.

---

## Project Structure

```
Proyecto_BDA_2/
|-- Team11C_data_collection.py        # Bronze layer: raw data ingestion
|-- Team11C_data_formatting.py        # Silver layer: cleaning and standardization
|-- Team11C_exploitation_zone.py      # Gold layer: feature engineering
|-- Team11C_ml_training.py            # ML pipeline (sequential)
|-- Team11C_ml_training_parallel.py   # ML pipeline (parallel)
|-- Team11C_airflow_setup.py          # Airflow initialization
|-- Team11C_start_airflow.sh          # Script to start Airflow
|-- Team11C_requirements.txt          # Python dependencies
|
|-- dags/
|   |-- Team11C_bda_rental_price_pipeline_optimized.py # Parallel DAG
|
|-- datasets/                 # Source data files
|   |-- idealista/            # Property listings (JSON)
|   |-- density/              # Population density (JSON)
|   |-- income/               # Income data (CSV)
|   |-- lookup_tables/        # Neighborhood ID mappings
|
|-- data_lake/
|   |-- bronze/               # Raw data (Parquet)
|   |-- silver/               # Cleaned data (CSV exports)
|   |-- gold/                 # ML-ready data (Delta Lake)
|
|-- airflow/                  # Airflow metadata and logs
|-- mlruns/                   # MLflow experiment tracking
|-- reports/                  # Generated plots and metrics
|-- logs/                     # Pipeline execution logs
```

---

## Troubleshooting

### Windows-Specific Issues

**Airflow fails with "No module named 'fcntl'" or "module 'signal' has no attribute 'SIGALRM'"**
- **Cause:** You're trying to run Airflow on native Windows, which is not supported.
- **Solution:** Use WSL2. See the Prerequisites section for installation instructions.

**"Cannot find Python" or "Command not found" in WSL2**
- Install Python in WSL2: `sudo apt update && sudo apt install python3 python3-pip python3-venv`

**MongoDB connection issues in WSL2**
- Install MongoDB in WSL2: Follow the [MongoDB WSL2 installation guide](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/)
- Start MongoDB: `sudo systemctl start mongod`

**File permission issues in WSL2**
- If working with files on `/mnt/c/`, they may have Windows permissions. Consider cloning the project in WSL2's home directory (`~/`) instead.

**"cannot execute: required file not found" when running .sh scripts in WSL2**
- **Cause:** The script has Windows line endings (CRLF) instead of Unix line endings (LF)
- **Solution:** Convert line endings using `dos2unix`:
  ```bash
  # Install dos2unix if not already installed
  sudo apt-get install dos2unix
  
  # Convert the script
  dos2unix Team11C_start_airflow.sh
  
  # Then run it
  ./Team11C_start_airflow.sh
  ```
- **Alternative:** Run with bash explicitly:
  ```bash
  bash Team11C_start_airflow.sh
  ```

### General Issues (All Platforms)

**MongoDB not connecting:** Make sure MongoDB is running (`sudo systemctl start mongod` or `brew services start mongodb-community`)

**Java not found:** Set `JAVA_HOME` to your Java installation path
- Linux/WSL2: `export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64`
- macOS: `export JAVA_HOME=$(/usr/libexec/java_home -v 11)`

**Port 8080 in use:** Kill the process using `lsof -i :8080` and `kill -9 <PID>`

**Out of memory:** Increase Spark memory with `export SPARK_DRIVER_MEMORY=4g`

**Check logs for detailed error messages:**
- Airflow logs: `airflow/logs/`
- Pipeline logs: `logs/`

---

## Acknowledgments

This project was developed as part of the Bases de Dades Avançades (BDA) course at FIB-UPC, Fall 2025.
