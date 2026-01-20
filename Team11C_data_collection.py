"""
Team11C_data_collection.py
---------------------
Data Ingestion Module for the Barcelona Rental Price Estimation Project.

This script handles the ingestion of raw data into the Bronze Layer (Landing Zone) of the Data Lake.
It implements an incremental loading strategy:
1. Identifies new files in the source directories that haven't been processed yet.
2. Reads these raw files (JSON, CSV).
3. Enriches the data with ingestion metadata (timestamp, source filename).
4. Writes the data to the 'data_lake/bronze' directory in Parquet format, partitioned by ingestion date.
5. Logs processed files to avoid duplication in future runs.

Usage:
    # Incremental loading (default - only new files)
    python Team11C_data_collection.py --collectors all
    python Team11C_data_collection.py --collectors idealista density
    
    # Reset and reload all data (delete Bronze layer and tracking logs)
    python Team11C_data_collection.py --reset
    python Team11C_data_collection.py --collectors idealista --reset
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, input_file_name, regexp_extract
import os
import sys
from datetime import datetime
import glob
import argparse

class DataCollector:
    """
    Base class for Data Collectors.
    
    Attributes:
        spark (SparkSession): The active Spark session.
        source_pattern (str): Glob pattern to match source files.
        destination_path (str): Target path in the Data Lake (Bronze Layer).
        tracking_file (str): Path to the log file tracking processed files.
    """
    def __init__(self, spark, source_pattern, destination_path, tracking_file):
        self.spark = spark
        self.source_pattern = source_pattern
        self.destination_path = destination_path
        self.tracking_file = tracking_file

    def get_new_files(self):
        """
        Identifies files that have not been processed yet.
        
        This implements incremental loading by comparing all source files
        against a tracking log of previously processed files.
        
        Returns:
            list: File paths that are new and need to be processed.
        """
        # Get all files matching the pattern
        all_files = set(glob.glob(self.source_pattern))
        
        # Load processed files
        processed_files = set()
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, 'r') as f:
                processed_files = set(line.strip() for line in f)
        
        # Determine new files
        new_files = list(all_files - processed_files)
        return new_files

    def mark_files_as_processed(self, files):
        """
        Updates the tracking file with newly processed files.
        
        Args:
            files (list): List of file paths that have been successfully processed.
        """
        with open(self.tracking_file, 'a') as f:
            for file_path in files:
                f.write(f"{file_path}\n")
    
    def reset_bronze_layer(self):
        """
        Resets the Bronze layer for this collector:
        1. Deletes the tracking log (so all files are considered "new")
        2. Deletes the Bronze layer destination directory
        
        This allows complete reprocessing of all source files.
        """
        import shutil
        
        # Remove tracking file
        if os.path.exists(self.tracking_file):
            os.remove(self.tracking_file)
            print(f"  Deleted tracking log: {self.tracking_file}")
        else:
            print(f"  No tracking log found: {self.tracking_file}")
        
        # Remove Bronze layer data
        if os.path.exists(self.destination_path):
            shutil.rmtree(self.destination_path)
            print(f"  Deleted Bronze layer data: {self.destination_path}")
        else:
            print(f"  No Bronze layer data found: {self.destination_path}")

    def collect(self):
        raise NotImplementedError

class IdealistaCollector(DataCollector):
    """
    Collector for Idealista property data (real estate listings).
    
    Input: JSON files from datasets/idealista/ directory
    Format: Multi-line JSON with property attributes (price, size, location, etc.)
    
    Transformations:
        - Adds ingestion_timestamp for audit trail
        - Extracts source_file for data lineage tracking
        - Extracts data_year from filename (YYYY_MM_DD format) for temporal filtering
        - Partitions by ingestion_date for efficient incremental loading
    
    Output: Parquet files in data_lake/bronze/idealista/
    """
    def collect(self):
        print(f"Checking for new Idealista data in {self.source_pattern}...")
        new_files = self.get_new_files()
        
        if not new_files:
            print("No new Idealista files to process.")
            return

        print(f"Found {len(new_files)} new files. Processing...")
        
        # Read ONLY new JSON files
        # spark.read.json accepts a list of paths
        df = self.spark.read.option("recursiveFileLookup", "true").json(new_files)
        
        # Add metadata: Ingestion time and source filename
        df_enriched = df.withColumn("ingestion_timestamp", current_timestamp()) \
                        .withColumn("source_file", input_file_name())
        
        # Extract year from filename (format: YYYY_MM_DD_idealista.json)
        # Using regex to extract the first 4 digits from the filename
        df_enriched = df_enriched.withColumn(
            "data_year",
            regexp_extract(input_file_name(), r"(\d{4})_\d{2}_\d{2}_idealista\.json", 1)
        )
        
        # Periodic Execution: Partition by ingestion date
        today = datetime.now().strftime("%Y-%m-%d")
        output_path = os.path.join(self.destination_path, f"ingestion_date={today}")
        
        # Write to Bronze (Parquet)
        df_enriched.write.mode("append").parquet(output_path)
        print(f"Idealista data written to {output_path}")
        
        # Update tracking log
        self.mark_files_as_processed(new_files)

class IncomeCollector(DataCollector):
    """
    Collector for Income data.
    Reads CSV files, adds metadata, and writes to Bronze layer in Parquet format.
    """
    def collect(self):
        print(f"Checking for new Income data in {self.source_pattern}...")
        new_files = self.get_new_files()
        
        if not new_files:
            print("No new Income files to process.")
            return

        print(f"Found {len(new_files)} new files. Processing...")

        # Read ONLY new CSV files
        df = self.spark.read.option("header", "true") \
                            .option("recursiveFileLookup", "true") \
                            .csv(new_files)
        
        df_enriched = df.withColumn("ingestion_timestamp", current_timestamp()) \
                        .withColumn("source_file", input_file_name())
        
        today = datetime.now().strftime("%Y-%m-%d")
        output_path = os.path.join(self.destination_path, f"ingestion_date={today}")
        
        df_enriched.write.mode("append").parquet(output_path)
        print(f"Income data written to {output_path}")
        
        self.mark_files_as_processed(new_files)

class DensityCollector(DataCollector):
    """
    Collector for Population Density data.
    Reads JSON files (handling multiline), adds metadata, and writes to Bronze layer in Parquet format.
    """
    def collect(self):
        print(f"Checking for new Density data in {self.source_pattern}...")
        new_files = self.get_new_files()
        
        if not new_files:
            print("No new Density files to process.")
            return

        print(f"Found {len(new_files)} new files. Processing...")

        # Read ONLY new JSON files
        df = self.spark.read.option("multiLine", "true") \
                            .option("recursiveFileLookup", "true") \
                            .json(new_files)
        
        df_enriched = df.withColumn("ingestion_timestamp", current_timestamp()) \
                        .withColumn("source_file", input_file_name())
        
        today = datetime.now().strftime("%Y-%m-%d")
        output_path = os.path.join(self.destination_path, f"ingestion_date={today}")
        
        df_enriched.write.mode("append").parquet(output_path)
        print(f"Density data written to {output_path}")
        
        self.mark_files_as_processed(new_files)

def run_collectors(selected_collectors=None, reset=False):
    """
    Orchestrates the data collection process.
    Args:
        selected_collectors (list): List of collector names to run. If None or ['all'], runs all.
        reset (bool): If True, deletes Bronze layer data and tracking logs before reprocessing.
    """
    # Initialize Spark Session
    # Set SPARK_LOCAL_IP to avoid "hostname resolves to loopback" warning
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
    
    # Suppress stderr (fd 2) and stdout (fd 1) to hide Spark/Ivy startup logs
    stderr_fd = sys.stderr.fileno()
    stdout_fd = sys.stdout.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    saved_stdout_fd = os.dup(stdout_fd)

    with open(os.devnull, 'w') as devnull:
        os.dup2(devnull.fileno(), stderr_fd)
        os.dup2(devnull.fileno(), stdout_fd)
        try:
            spark = SparkSession.builder \
                .appName("DataCollector") \
                .master("local[*]") \
                .config("spark.sql.shuffle.partitions", "8") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.default.parallelism", "4") \
                .getOrCreate()
            
            # Set log level to ERROR to suppress warnings
            spark.sparkContext.setLogLevel("ERROR")
        finally:
            os.dup2(saved_stderr_fd, stderr_fd)
            os.dup2(saved_stdout_fd, stdout_fd)
            os.close(saved_stderr_fd)
            os.close(saved_stdout_fd)
    
    # Define paths
    # Source: Local datasets folder
    base_source = "datasets"
    # Destination: Bronze Layer in Data Lake
    base_dest = "data_lake/bronze"
    # Tracking logs directory
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Instantiate Collectors with tracking files
    collectors = {
        "idealista": IdealistaCollector(
            spark, 
            f"{base_source}/idealista/*.json", 
            f"{base_dest}/idealista",
            f"{log_dir}/processed_idealista.log"
        ),
        "income": IncomeCollector(
            spark, 
            f"{base_source}/income/*.csv", 
            f"{base_dest}/income",
            f"{log_dir}/processed_income.log"
        ),
        "density": DensityCollector(
            spark, 
            f"{base_source}/density/*.json", 
            f"{base_dest}/density",
            f"{log_dir}/processed_density.log"
        )
    }
    
    # Determine which collectors to run
    if not selected_collectors or "all" in selected_collectors:
        to_run = collectors.values()
    else:
        to_run = [collectors[name] for name in selected_collectors if name in collectors]

    # Reset Bronze layer if requested
    if reset:
        print("RESETTING BRONZE LAYER (DELETE & RELOAD)")
        for collector in to_run:
            collector_name = type(collector).__name__
            print(f"\nResetting {collector_name}...")
            collector.reset_bronze_layer()
        print("Reset complete. Starting fresh data collection...")

    # Execute Collections
    for collector in to_run:
        collector.collect()
    
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Data Collectors")
    parser.add_argument(
        "--collectors", 
        nargs="+", 
        default=["all"], 
        choices=["idealista", "income", "density", "all"],
        help="Specify which collectors to run (space separated). Default is 'all'."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete Bronze layer data and tracking logs, then reload all source files from scratch."
    )
    
    args = parser.parse_args()
    run_collectors(args.collectors, reset=args.reset)
