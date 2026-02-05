"""
data_formatting.py
---------------------
Data Transformation Module (Bronze (Landing Zone) -> Silver (Formatted Zone)).

This script processes raw data from the Bronze Layer and loads it into the Silver Layer (MongoDB).
It performs the following operations:
1. Reads Parquet files from the Bronze layer.
2. Standardizes schemas (renaming columns, fixing data types).
3. Enriches data with standardized IDs using Lookup Tables (e.g., neighborhood_id).
4. Handles duplicates and ensures data quality.
5. Writes the cleaned and enriched records to MongoDB collections.
6. Exports CSV snapshots for manual inspection/Data Wrangler.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, first
import pymongo
from pymongo import UpdateOne
import os
import sys
import datetime
import argparse



YEARS_OF_INTEREST = [] # Populated dynamically during Idealista processing. Used to filter Income and Density data.

def create_mongodb_indexes(auto_recover=True):
    """
    Creates indexes on MongoDB collections for efficient upsert operations.
    Implements auto-recovery: if index creation fails due to duplicates, 
    automatically cleans the collection and retries.
    
    Indexes:
    - idealista: Unique index on 'property_id'
    - income: Compound index on ('neighborhood_id', 'year')
    - density: Compound index on ('neighborhood_id', 'year')
    
    Args:
        auto_recover: If True, automatically fix duplicate data issues
    """
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["bda_project"]
    
    index_definitions = [
        {
            "collection": "idealista",
            "keys": "property_id",
            "unique": True,
            "partial": None
        },
        {
            "collection": "income",
            "keys": [("neighborhood_id", 1), ("year", 1)],
            "unique": True,
            "partial": {"neighborhood_id": {"$exists": True, "$type": "string"}}
        },
        {
            "collection": "density",
            "keys": [("neighborhood_id", 1), ("year", 1)],
            "unique": True,
            "partial": {"neighborhood_id": {"$exists": True, "$type": "string"}}
        }
    ]
    
    for idx_def in index_definitions:
        collection = db[idx_def["collection"]]
        
        try:
            # Try to create index
            if idx_def["partial"]:
                collection.create_index(
                    idx_def["keys"],
                    unique=idx_def["unique"],
                    partialFilterExpression=idx_def["partial"]
                )
            else:
                collection.create_index(idx_def["keys"], unique=idx_def["unique"])
            
            print(f"Index created for {idx_def['collection']}")
            
        except pymongo.errors.DuplicateKeyError as e:
            # Duplicate data exists - auto-recover if enabled
            if auto_recover:
                print(f"Duplicate data detected in {idx_def['collection']}")
                print(f"Auto-recovering: Dropping collection and indexes...")
                
                # Drop the problematic collection entirely
                collection.drop()
                print(f"  Collection {idx_def['collection']} reset")
                
                # Recreate the index (now on empty collection)
                if idx_def["partial"]:
                    collection.create_index(
                        idx_def["keys"],
                        unique=idx_def["unique"],
                        partialFilterExpression=idx_def["partial"]
                    )
                else:
                    collection.create_index(idx_def["keys"], unique=idx_def["unique"])
                
                print(f"  Index recreated for {idx_def['collection']}")
            else:
                print(f"Index creation failed for {idx_def['collection']}: {e}")
                print(f"Run: venv/bin/python reset_mongodb.py")
                raise
        
        except pymongo.errors.OperationFailure as e:
            # Index already exists with different spec or other issue
            if "already exists" in str(e) or "existing index" in str(e).lower():
                print(f"Index already exists for {idx_def['collection']} (OK)")
            else:
                print(f"Index issue for {idx_def['collection']}: {e}")
    
    client.close()
    print("\nMongoDB indexes verified/created successfully.\n")

def write_to_mongo_upsert(partition, collection_name, unique_keys):
    """
    Writes a Spark partition to MongoDB using UPSERT (Update or Insert) strategy.
    
    Args:
        partition: Iterator over rows in the partition.
        collection_name: Target MongoDB collection.
        unique_keys: List of field names that form the unique key (e.g., ['property_id'] or ['neighborhood_id', 'year']).
        
    Process:
    1. Connects to MongoDB (localhost:27017).
    2. Converts rows to dicts and fixes date objects for Mongo compatibility.
    3. Creates UpdateOne operations with upsert=True.
    4. Executes in batches of 5000 for optimal performance.
    """
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["bda_project"]
        collection = db[collection_name]
        operations = []
        
        for row in partition:
            row_dict = row.asDict()
            
            # Convert datetime.date objects to datetime.datetime for Mongo compatibility
            for key, value in row_dict.items():
                if hasattr(value, 'strftime') and not hasattr(value, 'hour'):
                     row_dict[key] = datetime.datetime.combine(value, datetime.datetime.min.time())

            # Build query filter from unique keys
            query_filter = {key: row_dict[key] for key in unique_keys if key in row_dict}
            
            # Create upsert operation: Update if exists, Insert if not
            operations.append(
                UpdateOne(
                    query_filter,           # Match criteria
                    {"$set": row_dict},     # Update/Insert data
                    upsert=True             # Insert if not found
                )
            )
            
            # Execute in batches for performance
            if len(operations) >= 5000:
                collection.bulk_write(operations, ordered=False)
                operations = []
        
        # Execute remaining operations
        if operations:
            collection.bulk_write(operations, ordered=False)
            
        client.close()
    except Exception as e:
        print(f"Error writing to Mongo: {e}")

def process_idealista(spark):
    """
    Cleans and loads Idealista data into MongoDB using UPSERT strategy.
    
    Steps:
    1. Read Bronze Parquet.
    2. Select relevant columns and rename 'propertyCode' to 'property_id'.
    3. Deduplicate by 'property_id'.
    4. Upsert to 'idealista' collection (updates existing, inserts new).
    5. Save as CSV for Data Wrangler exploration.
    
    Unique Key: property_id
    """
    print("Processing Idealista...")
    try:
        # Read from Bronze Layer (Parquet)
        df = spark.read.parquet("data_lake/bronze/idealista")

        # Load Lookup Table and cache it (small table ~100 rows, reused frequently)
        lookup_df = spark.read.option("header", "true").csv("datasets/lookup_tables/idealista_extended.csv")
        lookup_df.cache()  # Cache for efficient broadcast join

        # Prepare Lookup: Keep only neighborhood (key) and neighborhood_id (target) to avoid collisions
        lookup_subset = lookup_df.select("neighborhood", "neighborhood_id")

        # Join with Lookup Table to get neighborhood_id (broadcast small lookup table)
        from pyspark.sql.functions import broadcast
        df = df.join(broadcast(lookup_subset), on="neighborhood", how="left")

        # Cache after join since we'll use df for both years extraction and final processing
        df.cache()
        
        # Get distinct years present in the data (triggers cache materialization)
        years = df.select("data_year").distinct().rdd.flatMap(lambda x: x).collect()
        YEARS_OF_INTEREST.extend(years)

        # Select and rename columns with explicit type casting
        df_clean = df.select(
            col("propertyCode").alias("property_id"),
            col("propertyType"),
            col("price").cast("double"),
            col("size").cast("double"),
            col("rooms").cast("integer"),
            col("bathrooms").cast("integer"),
            col("municipality"),
            col("neighborhood"),
            col("neighborhood_id"),  # From lookup table
            col("district"),
            col("latitude").cast("double"),
            col("longitude").cast("double"),
            col("hasLift").cast("boolean"),
            col("exterior").cast("boolean"),
            col("status"),
            col("distance").cast("double"),
            col("data_year").cast("integer")
        ).dropDuplicates(["property_id"])

        # Write to Mongo using UPSERT (idempotent)
        df_clean.foreachPartition(lambda p: write_to_mongo_upsert(p, "idealista", ["property_id"]))

        # Save as CSV for Data Wrangler exploration
        csv_output_path = "data_lake/silver/idealista_clean.csv"
        df_clean.coalesce(1).write.mode("overwrite").option("header", "true").csv(csv_output_path)
        print(f"Idealista processed and saved to {csv_output_path}")
        
        # Unpersist cached DataFrames to free memory for next task
        df.unpersist()
        lookup_df.unpersist()
    except Exception as e:
        print(f"Error processing Idealista: {e}")

def process_income(spark):
    """
    Formats and loads Income data into MongoDB using UPSERT strategy.
    
    Steps:
    1. Read Bronze Parquet.
    2. Rename Catalan columns to English (e.g., 'Nom_Barri' -> 'neighborhood_name').
    3. Upsert to 'income' collection (composite key: neighborhood_id + year).
    4. Save as CSV for Data Wrangler exploration.
    
    Unique Key: (neighborhood_id, year)
    """
    print("Processing Income...")
    try:
        # Read from Bronze Layer (Parquet)
        df = spark.read.parquet("data_lake/bronze/income")

        # Filter for years of interest
        df = df.filter(col("Any").isin(YEARS_OF_INTEREST))

        # Load Lookup Table and cache 
        lookup_df = spark.read.option("header", "true").csv("datasets/lookup_tables/income_opendatabcn_extended.csv")
        lookup_df.cache()  # Cache for efficient broadcast join

        # Join with Lookup Table to get neighborhood_id 
        # The lookup table has 'neighborhood' column which matches 'Nom_Barri' in income data
        from pyspark.sql.functions import broadcast
        df = df.join(broadcast(lookup_df), df["Nom_Barri"] == lookup_df["neighborhood"], how="left")

        # Rename columns to English/Standard and cast types
        df_clean = df.select(
            col("Any").cast("integer").alias("year"),
            col("Codi_Districte").alias("district_id"),
            col("Nom_Districte").alias("district_name"),
            col("Codi_Barri").alias("neighborhood_id_old"), # Keep original ID just in case
            col("neighborhood_id"), # ID from lookup
            col("Nom_Barri").alias("neighborhood_name"),
            col("Import_Euros").cast("double").alias("income"),
        )

        # Filter out records with NULL neighborhood_id (lookup failures)
        # These records cannot be joined with properties in exploitation zone
        df_clean = df_clean.filter(col("neighborhood_id").isNotNull())
        
        # Deduplicate by (neighborhood_id, year) - keep first occurrence or aggregate
        # If there are true duplicates in source data, take the average income
        df_clean = df_clean.groupBy("neighborhood_id", "year").agg(
            first("district_id").alias("district_id"),
            first("district_name").alias("district_name"),
            first("neighborhood_id_old").alias("neighborhood_id_old"),
            first("neighborhood_name").alias("neighborhood_name"),
            avg("income").alias("income")  # Average if multiple values exist
        )

        # Write to Mongo using UPSERT with composite key (idempotent)
        df_clean.foreachPartition(lambda p: write_to_mongo_upsert(p, "income", ["neighborhood_id", "year"]))

        # Save as CSV for Data Wrangler exploration
        csv_output_path = "data_lake/silver/income_clean.csv"
        df_clean.coalesce(1).write.mode("overwrite").option("header", "true").csv(csv_output_path)
        print(f"Income processed and saved to {csv_output_path}")
        
        # Unpersist cached lookup table to free memory
        lookup_df.unpersist()
    except Exception as e:
        print(f"Error processing Income: {e}")

def process_density(spark):
    """
    Loads Density data into MongoDB using UPSERT strategy.
    
    Steps:
    1. Read Bronze Parquet.
    2. Rename columns to English/Standard.
    3. Upsert to 'density' collection (composite key: neighborhood_id + year).
    4. Save as CSV for Data Wrangler exploration.
    
    Unique Key: (neighborhood_id, year)
    """
    print("Processing Density...")
    try:
        # Read from Bronze Layer (Parquet)
        df = spark.read.parquet("data_lake/bronze/density")

        # Filter for years of interest
        df = df.filter(col("Any").isin(YEARS_OF_INTEREST))

        # Load Lookup Table and cache (same table as income, ~73 neighborhoods)
        lookup_df = spark.read.option("header", "true").csv("datasets/lookup_tables/income_opendatabcn_extended.csv")
        lookup_df.cache()  # Cache for efficient broadcast join

        # Prepare Lookup: Keep only neighborhood (key) and neighborhood_id (target)
        # Note: In lookup, the column is 'neighborhood', in density it is 'Nom_Barri'
        lookup_subset = lookup_df.select(col("neighborhood").alias("lookup_neigh"), col("neighborhood_id"))

        # Join with Lookup Table to get neighborhood_id (broadcast small lookup table)
        from pyspark.sql.functions import broadcast
        df = df.join(broadcast(lookup_subset), df["Nom_Barri"] == lookup_subset["lookup_neigh"], how="left")

        # Rename columns to English/Standard and cast types
        df_clean = df.select(
            col("Any").cast("integer").alias("year"),
            col("Densitat (hab/ha)").cast("double").alias("density_val"),
            col("neighborhood_id") # ID from lookup
        )

        # Filter out records with NULL neighborhood_id (lookup failures)
        # These records cannot be joined with properties in exploitation zone
        df_clean = df_clean.filter(col("neighborhood_id").isNotNull())
        
        # Deduplicate by (neighborhood_id, year) - take average density if duplicates exist
        from pyspark.sql.functions import avg
        df_clean = df_clean.groupBy("neighborhood_id", "year").agg(
            avg("density_val").alias("density_val")
        )

        # Write to Mongo using UPSERT with composite key (idempotent)
        df_clean.foreachPartition(lambda p: write_to_mongo_upsert(p, "density", ["neighborhood_id", "year"]))

        # Save as CSV for Data Wrangler exploration (drop complex nested columns)
        csv_output_path = "data_lake/silver/density_clean.csv"
        df_clean.coalesce(1).write.mode("overwrite").option("header", "true").csv(csv_output_path)
        print(f"Density processed and saved to {csv_output_path}")
        
        # Unpersist cached lookup table to free memory
        lookup_df.unpersist()
    except Exception as e:
        print(f"Error processing Density: {e}")

def run_formatting(selected_tasks=None):
    """
    Orchestrates the data formatting process.
    Args:
        selected_tasks (list): List of formatting tasks to run. If None or ['all'], runs all.
    """
    # Set SPARK_LOCAL_IP to avoid "hostname resolves to loopback" warning
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'

    # Ensure PySpark workers use the same Python environment as the driver (the venv)
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
            spark = SparkSession.builder \
                .appName("DataFormatting") \
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

    # Create MongoDB indexes for efficient upsert operations (one-time, idempotent)
    create_mongodb_indexes()
    
    tasks = {
        "idealista": process_idealista,
        "income": process_income,
        "density": process_density
    }

    # Determine which tasks to run
    if not selected_tasks or "all" in selected_tasks:
        to_run = tasks.values()
    else:
        to_run = [tasks[name] for name in selected_tasks if name in tasks]

    # Execute Tasks
    for task in to_run:
        task(spark)

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Data Formatting Tasks")
    parser.add_argument(
        "--tasks", 
        nargs="+", 
        default=["all"], 
        choices=["idealista", "income", "density", "all"],
        help="Specify which formatting tasks to run (space separated). Default is 'all'."
    )

    args = parser.parse_args()
    run_formatting(args.tasks)