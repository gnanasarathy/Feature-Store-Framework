# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Store Setup Notebook
# MAGIC
# MAGIC This notebook creates feature tables in Databricks Feature Store from CSV files.
# MAGIC
# MAGIC ## Requirements:
# MAGIC - Source folder containing: customer_features.csv, product_features.csv, training_labels.csv, inference_data.csv
# MAGIC - Databricks Feature Engineering package
# MAGIC
# MAGIC ## Parameters:
# MAGIC - **Source Folder Path**: Path to folder containing source CSV files

# COMMAND ----------

# Create widget for source file path
# dbutils.widgets.text("Source Folder Path", "", "Source Folder Path")

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Import required libraries
import logging
from typing import Dict, List, Optional
from databricks.feature_store import FeatureStoreClient
from pyspark.sql import DataFrame
from pyspark.sql.utils import AnalysisException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration and Validation

# COMMAND ----------

# Configuration constants
DATABASE_NAME = "sales"
EXPECTED_FILES = [
    "customer_features.csv", 
    "product_features.csv", 
    "training_labels.csv", 
    "inference_data.csv"
]

FEATURE_TABLES_CONFIG = {
    "customer_features": {
        "primary_keys": ["customer_id"], 
        "file_name": "customer_features.csv",
        "description": "Customer feature table containing customer demographics and preferences"
    }, 
    "product_features": {
        "primary_keys": ["product_id"], 
        "file_name": "product_features.csv",
        "description": "Product feature table containing product characteristics and metadata"
    }
}

# CSV read options
CSV_OPTIONS = {
    "format": "csv",
    "sep": ",",
    "inferSchema": "true",
    "header": "true"
}

# COMMAND ----------

def validate_source_path(source_path: str) -> None:
    """Validate that source path is provided and accessible."""
    if not source_path or source_path.strip() == "":
        raise ValueError("Source folder path is required but not provided")
    
    try:
        dbutils.fs.ls(source_path)
        logger.info(f"Source path validated: {source_path}")
    except Exception as e:
        raise FileNotFoundError(f"Cannot access source path '{source_path}': {str(e)}")

def validate_required_files(source_path: str, required_files: list[str]) -> None:
    """Check if all required files exist in the source folder."""
    try:
        existing_files = dbutils.fs.ls(source_path)
        existing_filenames = [file.name for file in existing_files]
        
        missing_files = [file for file in required_files if file not in existing_filenames]
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")
        
        logger.info(f"All required files found: {', '.join(required_files)}")
        
    except Exception as e:
        if "Missing required files" in str(e):
            raise
        else:
            raise Exception(f"Error checking files in source path: {str(e)}")

def create_database_if_not_exists(database_name: str) -> None:
    """Create database if it doesn't exist."""
    try:
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {database_name}")
        logger.info(f"Database '{database_name}' is ready")
    except Exception as e:
        raise Exception(f"Failed to create database '{database_name}': {str(e)}")

# COMMAND ----------

# Get and validate source folder path
source_folder_path = dbutils.widgets.get("Source Folder Path").strip()
try:
    validate_source_path(source_folder_path)
    validate_required_files(source_folder_path, EXPECTED_FILES)
    create_database_if_not_exists(DATABASE_NAME)
except Exception as e:
    logger.error(f"Validation failed: {str(e)}")
    dbutils.notebook.exit(f"ERROR: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Table Creation Functions

# COMMAND ----------

def load_csv_data(file_path: str) -> DataFrame:
    """Load CSV file and return DataFrame with basic validation."""
    try:
        df = spark.read.load(file_path, **CSV_OPTIONS)
        
        # Basic validation
        if df.count() == 0:
            raise ValueError(f"File {file_path} is empty")
        
        logger.info(f"Successfully loaded {df.count()} rows from {file_path}")
        logger.info(f"Schema: {df.columns}")
        
        return df
        
    except Exception as e:
        raise Exception(f"Failed to load data from {file_path}: {str(e)}")

def drop_existing_table(fs_client: FeatureStoreClient, table_name: str) -> None:
    """Safely drop existing feature table if it exists."""
    try:
        fs_client.drop_table(name=table_name)
        logger.info(f"Dropped existing table: {table_name}")
    except Exception as e:
        # Table might not exist, which is fine
        logger.info(f"No existing table to drop for {table_name} (or error occurred): {str(e)}")

def create_feature_table(
    fs_client: FeatureStoreClient, 
    table_name: str, 
    primary_keys: List[str], 
    df: DataFrame, 
    description: str
) -> None:
    """Create a new feature table."""
    try:
        # Validate primary keys exist in DataFrame
        missing_keys = [key for key in primary_keys if key not in df.columns]
        if missing_keys:
            raise ValueError(f"Primary keys {missing_keys} not found in DataFrame columns: {df.columns}")
        
        # Create feature table
        fs_client.create_table(
            name=table_name,
            primary_keys=primary_keys,
            df=df,
            schema=df.schema,
            description=description
        )
        
        logger.info(f"Successfully created feature table: {table_name}")
        logger.info(f"  - Primary keys: {primary_keys}")
        logger.info(f"  - Columns: {df.columns}")
        logger.info(f"  - Row count: {df.count()}")
        
    except Exception as e:
        raise Exception(f"Failed to create feature table {table_name}: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Feature Tables

# COMMAND ----------

# Initialize Feature Store Client
try:
    fs = FeatureStoreClient()
    logger.info("Feature Store Client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Feature Store Client: {str(e)}")
    dbutils.notebook.exit(f"ERROR: Failed to initialize Feature Store Client: {str(e)}")

# Create feature tables
successful_tables = []
failed_tables = []

for table_key, config in FEATURE_TABLES_CONFIG.items():
    table_name = f"{DATABASE_NAME}.{table_key}"
    
    try:
        logger.info(f"Processing feature table: {table_name}")
        
        # Load data
        file_path = f"{source_folder_path}/{config['file_name']}"
        df = load_csv_data(file_path)
        
        # Drop existing table
        drop_existing_table(fs, table_name)
        
        # Create new table
        create_feature_table(
            fs_client=fs,
            table_name=table_name,
            primary_keys=config["primary_keys"],
            df=df,
            description=config["description"]
        )
        
        successful_tables.append(table_name)
        
    except Exception as e:
        error_msg = f"Failed to create table {table_name}: {str(e)}"
        logger.error(error_msg)
        failed_tables.append((table_name, str(e)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Report

# COMMAND ----------

# Print summary
print("=" * 60)
print("FEATURE TABLE CREATION SUMMARY")
print("=" * 60)

if successful_tables:
    print(f"✅ Successfully created {len(successful_tables)} feature table(s):")
    for table in successful_tables:
        print(f"   - {table}")

if failed_tables:
    print(f"\n❌ Failed to create {len(failed_tables)} feature table(s):")
    for table, error in failed_tables:
        print(f"   - {table}: {error}")

print(f"\nSource folder: {source_folder_path}")
print(f"Database: {DATABASE_NAME}")
print("=" * 60)

# Exit with error if any tables failed
if failed_tables:
    error_summary = f"Failed to create {len(failed_tables)} out of {len(FEATURE_TABLES_CONFIG)} feature tables"
    dbutils.notebook.exit(f"ERROR: {error_summary}")
else:
    print("✅ All feature tables created successfully!")
