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


