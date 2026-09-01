# ==============================================================================
# us_visa/entity/config_entity.py
# INPUT configuration blueprints for every pipeline component
# ==============================================================================
import os
from us_visa.constants import *
from dataclasses import dataclass
from datetime import datetime


TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# ------------------------------------------------------------------------------
# 0. MASTER PIPELINE CONFIG (creates the timestamped artifact root)
# ------------------------------------------------------------------------------
@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir : str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP

# Single global instance shared by all component configs
training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

# ------------------------------------------------------------------------------
# 1. DATA INGESTION CONFIG
# ------------------------------------------------------------------------------
@dataclass
class DataIngestionConfig:
    """Configuration for the data ingestion component."""
     # Root: artifacts/<timestamp>/data_ingestion/
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)

    # Raw data snapshot from MongoDB: .../feature_store/us_visa.csv
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)

    # Split outputs: .../ingested/train.csv and .../ingested/test.csv
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)

    # Split behaviour
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

    # MongoDB source
    collection_name: str = DATA_INGESTION_COLLECTION_NAME 


# ------------------------------------------------------------------------------
# 2. DATA VALIDATION CONFIG
# ------------------------------------------------------------------------------
@dataclass
class DataValidationConfig:
    """Inputs required by the DataValidation component."""
    data_validation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME
    )

    drift_report_file_path: str = os.path.join(
        data_validation_dir, DATA_VALIDATION_DRIFT_REPORT_DIR, DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
    )

# ------------------------------------------------------------------------------
# 3. DATA TRANSFORMATION CONFIG
# Converts flat constant names into full timestamped directory paths
# ------------------------------------------------------------------------------
@dataclass
class DataTransformationConfig:
    """
    Input configuration for the DataTransformation component.
    All paths are auto-generated inside the timestamped artifact root.

    Example resolved paths:
        data_transformation_dir        → artifacts/2026_03_08_17_00_01/data_transformation/
        transformed_train_file_path    → artifacts/.../data_transformation/transformed/train.npy
        transformed_test_file_path     → artifacts/.../data_transformation/transformed/test.npy
        transformed_object_file_path   → artifacts/.../data_transformation/transformed_object/preprocessor.pkl
    """
    # Root directory for all transformation artifacts
    data_transformation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,DATA_TRANSFORMATION_DIR_NAME
    )
    # Path to save the processed training numpy array
    transformed_train_file_path: str = os.path.join(
        data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DIR,TRAIN_FILE_NAME.replace("csv", "npy")
    )
    # Path to save the processed test numpy array
    transformed_test_file_path: str = os.path.join(
        data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DIR, TEST_FILE_NAME.replace("csv", "npy")
    )
    # Path to save the fitted ColumnTransformer object (for inference)
    transformed_object_file_path: str = os.path.join(
        data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR, PREPROCESSING_OBJECT_FILE_NAME
    )


# ==============================================================================
# 4. MODEL TRAINER CONFIGURATION
# ==============================================================================
@dataclass
class ModelTrainerConfig:
    """
    Configuration class for the Model Trainer component.

    This class defines:
        - Model Trainer artifact directory
        - Trained model file path
        - Minimum expected model performance
        - Model configuration YAML file path
    """

    # --------------------------------------------------------------------------
    # Model Trainer Artifact Directory
    # --------------------------------------------------------------------------
    # Example:
    # artifact/model_trainer
    model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        MODEL_TRAINER_DIR_NAME
    )

    # --------------------------------------------------------------------------
    # Trained Model File Path
    # --------------------------------------------------------------------------
    # Example:
    # artifact/model_trainer/trained_model/model.pkl
    trained_model_file_path: str = os.path.join(
        model_trainer_dir,
        MODEL_TRAINER_TRAINED_MODEL_DIR,
        MODEL_TRAINER_TRAINED_MODEL_NAME
    )

    # --------------------------------------------------------------------------
    # Minimum Expected Model Score
    # --------------------------------------------------------------------------
    # This threshold is used to determine whether the trained model
    # performs well enough to be accepted by the pipeline.
    expected_f1_score: float = MODEL_TRAINER_EXPECTED_SCORE

    # --------------------------------------------------------------------------
    # Model Configuration File
    # --------------------------------------------------------------------------
    # Points to config/model.yaml, which contains:
    #     - Model definitions
    #     - Default parameters
    #     - Hyperparameter search spaces
    #     - GridSearchCV configuration
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH


# ==============================================================================
# 5. MODEL EVALUATION CONFIG
# ==============================================================================

@dataclass
class ModelEvaluationConfig:
    """
    Configuration for the Model Evaluation component.

    Attributes:
        changed_threshold_score:
            Minimum score threshold used to determine whether the newly
            trained model is sufficiently better than the existing model.

        bucket_name:
            Name of the AWS S3 bucket where the model registry is stored.

        s3_model_key_path:
            S3 object key/path used to locate the model in the bucket.
    """

    # --------------------------------------------------------------------------
    # Model comparison threshold
    # --------------------------------------------------------------------------
    # Used when comparing the newly trained model against the currently
    # registered model.
    changed_threshold_score: float = (
        MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    )

    # --------------------------------------------------------------------------
    # AWS S3 bucket name
    # --------------------------------------------------------------------------
    # Identifies the S3 bucket containing the model registry.
    bucket_name: str = MODEL_BUCKET_NAME

    # --------------------------------------------------------------------------
    # S3 model key/path
    # --------------------------------------------------------------------------
    # Identifies the trained model file inside the S3 bucket.
    s3_model_key_path: str = MODEL_FILE_NAME



# ==============================================================================
# MODEL PUSHER CONFIGURATION
# ==============================================================================

@dataclass
class ModelPusherConfig:
    """
    Configuration settings for the Model Pusher component.

    The Model Pusher is responsible for uploading the accepted trained
    model from the local system to the configured AWS S3 model registry.
    """

    # --------------------------------------------------------------------------
    # AWS S3 bucket where the trained model will be stored.
    # --------------------------------------------------------------------------
    bucket_name: str = MODEL_BUCKET_NAME

    # --------------------------------------------------------------------------
    # S3 object key/path where the trained model will be uploaded.
    #
    # Example:
    #     model-registry/model.pkl
    # --------------------------------------------------------------------------
    s3_model_key_path: str = MODEL_FILE_NAME

