# ------------------------------------------------------------------------------
# 1. PROJECT ROOT & DIRECTORY STRUCTURE
# ------------------------------------------------------------------------------
import os
from datetime import date

DATABASE_NAME = "US_VISA_DB"
COLLECTION_NAME = "visa_applications"
MONGODB_URL_KEY = "MONGODB_URL"

PIPELINE_NAME: str = "usvisa"
ARTIFACT_DIR: str = "artifact"

MODEL_FILE_NAME = "model.pkl"

TARGET_COLUMN = "case_status"
CURRENT_YEAR = date.today().year
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

FILE_NAME: str = "usvisa.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


# ------------------------------------------------------------------------------
# 2. DATA INGESTION CONSTANTS (Directories Name)
# ------------------------------------------------------------------------------
"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "visa_applications"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# ------------------------------------------------------------------------------
# 3. DATA VALIDATION CONSTANTS
# ------------------------------------------------------------------------------
""" 
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

# ------------------------------------------------------------------------------
# 4. DATA TRANSFORMATION CONSTANTS
# ------------------------------------------------------------------------------
"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
# Directory name for transformation artifacts
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
# Sub-directory names
DATA_TRANSFORMATION_TRANSFORMED_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"



# ------------------------------------------------------------------------------
# 5. MODEL TRAINER CONSTANTS
# ------------------------------------------------------------------------------
"""
Constants related to the Model Trainer component.

All Model Trainer-related constants start with the
'MODEL_TRAINER' prefix for easy identification and maintenance.
"""
# ------------------------------------------------------------------------------
# Model Trainer Directory Configuration
# ------------------------------------------------------------------------------

# Main directory where Model Trainer artifacts will be stored.
MODEL_TRAINER_DIR_NAME: str = "model_trainer"

# Directory inside the Model Trainer artifact directory
# where the trained model will be saved.
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"

# File name of the final trained model.
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"


# ------------------------------------------------------------------------------
# Model Performance Configuration
# ------------------------------------------------------------------------------

# Minimum expected F1-score required for a model to be considered
# good enough for the next stage of the ML pipeline.
#
# Example:
#     If F1-score >= 0.60 → Model is accepted
#     If F1-score <  0.60 → Model is rejected
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6


# ------------------------------------------------------------------------------
# Model Configuration File
# ------------------------------------------------------------------------------

# Path to the YAML file containing:
#     - Model definitions
#     - Default model parameters
#     - Hyperparameter search spaces
#     - GridSearchCV configuration
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join(
    "config",
    "model.yaml"
)

# ==============================================================================
# 6. MODEL EVALUATION CONSTANTS
# ==============================================================================
"""
Constants related to the Model Evaluation component.

These constants define:
    1. The minimum improvement threshold required for a newly trained model.
    2. The AWS S3 bucket used for storing the model registry.
    3. The S3 directory/key where registered models are stored.
"""


# ------------------------------------------------------------------------------
# Minimum score improvement required for model replacement
# ------------------------------------------------------------------------------
# A newly trained model must improve over the currently registered model
# by at least this threshold before it can replace the existing model.
#
# Example:
#     Existing model score = 0.80
#     New model score      = 0.85
#     Improvement          = 0.05
#
#     Since 0.05 < 0.50, the new model would NOT replace the existing model.
#
# NOTE:
# The exact interpretation depends on how the Model Evaluation component
# compares the scores.
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.5


# ------------------------------------------------------------------------------
# AWS S3 Bucket Configuration
# ------------------------------------------------------------------------------
# Name of the AWS S3 bucket used as the model registry/storage location.
MODEL_BUCKET_NAME: str = "usvisabucke26"


# ------------------------------------------------------------------------------
# S3 Model Registry Directory
# ------------------------------------------------------------------------------
# S3 key/prefix used to organize registered models inside the bucket.
#
# Expected S3 structure:
#
#     usvisabucke26/
#     └── model-registry/
#         └── model.pkl
#
MODEL_PUSHER_S3_KEY: str = "model-registry"




