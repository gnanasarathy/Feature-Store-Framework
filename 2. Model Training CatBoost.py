# Databricks notebook source
# MAGIC %md
# MAGIC # ML Model Training with Feature Store
# MAGIC
# MAGIC This notebook trains a machine learning model using features from Databricks Feature Store.
# MAGIC
# MAGIC ## Requirements:
# MAGIC - Source folder containing training_labels.csv
# MAGIC - Existing feature tables: customer_features, product_features
# MAGIC - Required packages: databricks-feature-engineering, catboost
# MAGIC
# MAGIC ## Parameters:
# MAGIC - **Source Folder Path**: Path to folder containing training labels CSV file

# COMMAND ----------

# Create widget for source file path
# dbutils.widgets.text("Source Folder Path", "", "Source Folder Path")

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering
# MAGIC %pip install catboost

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Import required libraries
import logging
from typing import Dict, List, Tuple, Optional
import warnings
import pandas as pd
import numpy as np
from pyspark.sql import DataFrame

# ML libraries
import mlflow
import mlflow.sklearn
from mlflow.tracking.client import MlflowClient
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# Feature Store
from databricks.feature_store import FeatureStoreClient, FeatureLookup

# Configure logging and warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration constants
DATABASE_NAME = "sales"
MODEL_NAME = f"{DATABASE_NAME}.purchase_model"
LABEL_COLUMN = "purchased"
EXCLUDE_COLUMNS = ['customer_id', 'product_id']

# Feature lookups configuration for online store
FEATURE_LOOKUPS_CONFIG = [
    {
        'table_name': f'workspace.{DATABASE_NAME}.customer_features',
        'feature_names': ['total_purchase_7d', 'total_purchase_30d'],
        'lookup_key': 'customer_id',
        'lookup_mode': 'online'
    },
    {
        'table_name': f'workspace.{DATABASE_NAME}.product_features',
        'feature_names': ['category'],
        'lookup_key': 'product_id'
    }
]

# CatBoost model hyperparameters
MODEL_PARAMS = {
    'iterations': 100,
    'learning_rate': 0.1,
    'depth': 6,
    'random_seed': 42,
    'verbose': False,
    'eval_metric': 'Logloss',
    'od_type': 'Iter',
    'od_wait': 20,
    'use_best_model': True
}

# Train-test split parameters
SPLIT_PARAMS = {
    'test_size': 0.2,
    'random_state': 42,
    'stratify': None  # Will be set dynamically
}

# CSV read options
CSV_OPTIONS = {
    "format": "csv",
    "sep": ",",
    "inferSchema": "true",
    "header": "true"
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation Functions

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

def validate_label_file(file_path: str) -> None:
    """Validate that the training labels file exists."""
    try:
        dbutils.fs.ls(file_path)
        logger.info(f"Training labels file found: {file_path}")
    except Exception as e:
        raise FileNotFoundError(f"Training labels file not found: {file_path}")

def validate_feature_tables(fs_client: FeatureStoreClient, feature_lookups: List[FeatureLookup]) -> None:
    """Validate that all required feature tables exist."""
    for lookup in feature_lookups:
        try:
            # Try to read the table to verify it exists
            df = fs_client.read_table(lookup.table_name)
            logger.info(f"Feature table validated: {lookup.table_name}")
        except Exception as e:
            raise Exception(f"Feature table '{lookup.table_name}' not accessible: {str(e)}")

# COMMAND ----------

# Get and validate source folder path
source_folder_path = dbutils.widgets.get("Source Folder Path").strip()

try:
    validate_source_path(source_folder_path)
    label_file_path = f"{source_folder_path}/training_labels.csv"
    validate_label_file(label_file_path)
except Exception as e:
    logger.error(f"Validation failed: {str(e)}")
    dbutils.notebook.exit(f"ERROR: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Store Setup

# COMMAND ----------

def create_feature_lookups(config: List[Dict]) -> List[FeatureLookup]:
    """Create FeatureLookup objects from configuration."""
    feature_lookups = []
    
    for lookup_config in config:
        try:
            feature_lookup = FeatureLookup(
                table_name=lookup_config['table_name'],
                feature_names=lookup_config['feature_names'],
                lookup_key=lookup_config['lookup_key']
            )
            feature_lookups.append(feature_lookup)
            logger.info(f"Created feature lookup for {lookup_config['table_name']}")
            
        except Exception as e:
            raise Exception(f"Failed to create feature lookup for {lookup_config['table_name']}: {str(e)}")
    
    return feature_lookups

# Initialize Feature Store Client and create feature lookups
try:
    fs = FeatureStoreClient()
    feature_lookups = create_feature_lookups(FEATURE_LOOKUPS_CONFIG)
    
    # Validate feature tables exist
    validate_feature_tables(fs, feature_lookups)
    
    logger.info("Feature Store setup completed successfully")
    
except Exception as e:
    logger.error(f"Feature Store setup failed: {str(e)}")
    dbutils.notebook.exit(f"ERROR: Feature Store setup failed: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Loading and Preparation

# COMMAND ----------

def load_training_labels(file_path: str) -> DataFrame:
    """Load training labels from CSV file."""
    try:
        df = spark.read.load(file_path, **CSV_OPTIONS)
        
        # Basic validation
        if df.count() == 0:
            raise ValueError("Training labels file is empty")
        
        # Check if label column exists
        if LABEL_COLUMN not in df.columns:
            raise ValueError(f"Label column '{LABEL_COLUMN}' not found in training data")
        
        logger.info(f"Loaded {df.count()} training samples")
        logger.info(f"Columns: {df.columns}")
        
        return df
        
    except Exception as e:
        raise Exception(f"Failed to load training labels: {str(e)}")

def create_training_dataset(
    fs_client: FeatureStoreClient,
    label_df: DataFrame,
    feature_lookups: List[FeatureLookup],
    label_col: str,
    exclude_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, object]:
    """Create training dataset by joining labels with features."""
    
    try:
        # Create training set with feature lookups
        training_set = fs_client.create_training_set(
            df=label_df,
            feature_lookups=feature_lookups,
            label=label_col,
            exclude_columns=exclude_cols
        )
        
        # Convert to pandas
        training_pd = training_set.load_df().toPandas()
        
        logger.info(f"Training dataset shape: {training_pd.shape}")
        logger.info(f"Features: {[col for col in training_pd.columns if col != label_col]}")
        
        # Prepare features and target
        X = training_pd.drop(label_col, axis=1)
        y = training_pd[label_col]
        
        # Log class distribution
        class_counts = y.value_counts()
        logger.info(f"Class distribution: {dict(class_counts)}")
        
        # Create train-test split with stratification
        split_params = SPLIT_PARAMS.copy()
        split_params['stratify'] = y if len(class_counts) > 1 else None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, **split_params)
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test, training_set
        
    except Exception as e:
        raise Exception(f"Failed to create training dataset: {str(e)}")

# Load training labels
try:
    label_df = load_training_labels(label_file_path)
    display(label_df)
except Exception as e:
    logger.error(f"Failed to load training labels: {str(e)}")
    dbutils.notebook.exit(f"ERROR: {str(e)}")

# COMMAND ----------

# Create training dataset
try:
    X_train, X_test, y_train, y_test, training_set = create_training_dataset(
        fs_client=fs,
        label_df=label_df,
        feature_lookups=feature_lookups,
        label_col=LABEL_COLUMN,
        exclude_cols=EXCLUDE_COLUMNS
    )
    
    logger.info("Training dataset created successfully")
    display(X_train.head())
    
except Exception as e:
    logger.error(f"Failed to create training dataset: {str(e)}")
    dbutils.notebook.exit(f"ERROR: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preprocessing for CatBoost

# COMMAND ----------

def identify_categorical_features(X_train: pd.DataFrame) -> List[int]:
    """Identify categorical features for CatBoost."""
    
    categorical_features = []
    
    # Get categorical columns (object/string types)
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns
    
    # Get column indices for categorical features
    for col in categorical_columns:
        if col in X_train.columns:
            categorical_features.append(X_train.columns.get_loc(col))
    
    logger.info(f"Categorical columns: {list(categorical_columns)}")
    logger.info(f"Categorical feature indices: {categorical_features}")
    logger.info(f"Numerical columns: {list(X_train.select_dtypes(include=['number']).columns)}")
    
    return categorical_features

def preprocess_for_catboost(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    """Minimal preprocessing for CatBoost - it handles categorical features automatically."""
    
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    # Handle missing values if any
    if X_train_processed.isnull().any().any():
        logger.info("Found missing values - CatBoost will handle them automatically")
    
    # Identify categorical features for CatBoost
    categorical_features = identify_categorical_features(X_train_processed)
    
    logger.info("Minimal preprocessing completed - CatBoost will handle categorical features automatically")
    
    return X_train_processed, X_test_processed, categorical_features

# Preprocess features for CatBoost
try:
    X_train_processed, X_test_processed, categorical_features = preprocess_for_catboost(X_train, X_test)
    logger.info("Feature preprocessing completed")
    logger.info(f"Dataset shape: {X_train_processed.shape}")
    logger.info(f"Categorical features count: {len(categorical_features)}")
except Exception as e:
    logger.error(f"Feature preprocessing failed: {str(e)}")
    dbutils.notebook.exit(f"ERROR: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training and Evaluation

# COMMAND ----------

def calculate_metrics(y_true: pd.Series, y_pred: pd.Series, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics."""
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # Add AUC if probabilities are provided
    if y_proba is not None:
        try:
            if len(pd.Series(y_true).unique()) == 2:  # Binary classification
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
            else:  # Multi-class
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {str(e)}")
    
    return metrics

def clean_existing_model(client: MlflowClient, model_name: str) -> None:
    """Clean up existing registered model if it exists."""
    try:
        client.delete_registered_model(model_name)
        logger.info(f"Deleted existing model: {model_name}")
    except Exception:
        logger.info(f"No existing model to delete: {model_name}")

def train_and_evaluate_catboost_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    categorical_features: List[int],
    training_set: object,
    fs_client: FeatureStoreClient,
    model_params: Dict,
    model_name: str
) -> None:
    """Train CatBoost model and log to MLflow with comprehensive evaluation."""
    
    # Setup MLflow
    mlflow.sklearn.autolog(log_models=False)
    
    # Clean existing model
    client = MlflowClient()
    clean_existing_model(client, model_name)
    
    with mlflow.start_run() as run:
        try:
            logger.info("Starting CatBoost model training...")
            
            # Initialize CatBoost model with categorical features
            model = CatBoostClassifier(
                cat_features=categorical_features,
                **model_params
            )
            
            # Prepare evaluation set for early stopping
            eval_set = (X_test, y_test)
            
            # Train model with evaluation set
            model.fit(
                X_train, 
                y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, y_pred, y_proba)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)
                logger.info(f"Test {metric_name}: {metric_value:.4f}")
            
            # Log additional information
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_categorical_features", len(categorical_features))
            mlflow.log_param("best_iteration", model.get_best_iteration())
            
            # Log classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            mlflow.log_text(str(classification_report(y_test, y_pred)), "classification_report.txt")
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.get_feature_importance()
            }).sort_values('importance', ascending=False)
            
            mlflow.log_text(feature_importance.to_string(), "feature_importance.txt")
            logger.info("Top 5 important features:")
            logger.info(feature_importance.head().to_string())
            
            # Log CatBoost specific metrics
            if hasattr(model, 'get_best_score'):
                best_score = model.get_best_score()
                if 'validation' in best_score:
                    mlflow.log_metric("best_validation_score", best_score['validation']['Logloss'])
            
            # Log model with Feature Store
            fs_client.log_model(
                model=model,
                artifact_path="catboost_purchase_prediction_model",
                flavor=mlflow.sklearn,
                training_set=training_set,
                registered_model_name=model_name
            )
            
            logger.info(f"CatBoost model successfully registered as: {model_name}")
            logger.info(f"MLflow run ID: {run.info.run_id}")
            logger.info(f"Best iteration: {model.get_best_iteration()}")
            
        except Exception as e:
            logger.error(f"CatBoost model training failed: {str(e)}")
            raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute Training

# COMMAND ----------

# Train and evaluate CatBoost model
try:
    train_and_evaluate_catboost_model(
        X_train=X_train_processed,
        X_test=X_test_processed,
        y_train=y_train,
        y_test=y_test,
        categorical_features=categorical_features,
        training_set=training_set,
        fs_client=fs,
        model_params=MODEL_PARAMS,
        model_name=MODEL_NAME
    )
    
    print("=" * 60)
    print("✅ CATBOOST MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Model registered as: {MODEL_NAME}")
    print(f"Training samples: {len(X_train_processed)}")
    print(f"Test samples: {len(X_test_processed)}")
    print(f"Total features: {X_train_processed.shape[1]}")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Features used: {list(X_train_processed.columns)}")
    print("=" * 60)
    
except Exception as e:
    logger.error(f"CatBoost model training pipeline failed: {str(e)}")
    dbutils.notebook.exit(f"ERROR: CatBoost model training failed: {str(e)}")
