# ==============================================================================
# us_visa/pipeline/training_pipeline.py
# Master Orchestrator: Executes all ML pipeline components in sequence
# ==============================================================================
import sys
from us_visa.exception import USvisaException
from us_visa.logger import logging

from us_visa.components.data_ingestion import DataIngestion

from us_visa.entity.config_entity import(
    DataIngestionConfig
)

from us_visa.entity.artifact_entity import(
    DataIngestionArtifact
)


class TrainingPipeline:
    """
    Master orchestrator for the US Visa ML Pipeline.
    Each method corresponds to one pipeline stage.
    The run_pipeline() method executes all stages in sequence.
    """
    def __init__(self):
        """Initialize the master pipeline configuration."""
        self.data_ingestion_config = DataIngestionConfig()


    # ------------------------------------------------------------------
    # STAGE 1: DATA INGESTION
    # ------------------------------------------------------------------
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Executes the Data Ingestion component.

        Flow:
            DataIngestionConfig → DataIngestion(component) → DataIngestionArtifact

        Returns:
            DataIngestionArtifact containing paths to train.csv and test.csv.
        """
        try:
            logging.info(">>> STAGE 1: DATA INGESTION — Starting...")

            logging.info("Getting the data from mongodb")

            # Step 1 and 2: Create component configuration and Initialize component with config
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config                
            )
            
            
            # Step 3: Execute and receive artifact
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            

            logging.info("""
                >>> STAGE 1: DATA INGESTION — Completed
                    - Got the train_set and test_set from mongodb
                """)

            logging.info(
                "Exited the start_data_ingestion method of TrainingPipeline class"
            )
            return data_ingestion_artifact
        

        except Exception as e:
            raise USvisaException(e, sys) from e

        # ------------------------------------------------------------------
    # MASTER RUN: Execute all stages in sequence
    # ------------------------------------------------------------------
    def run_pipeline(self) -> None:
        """
        Executes the complete ML training pipeline from data ingestion
        to model deployment. Each stage receives the artifact from the
        previous stage as input.
        """
        try:
            logging.info("=" * 70)
            logging.info("US VISA ML TRAINING PIPELINE — EXECUTION STARTED")
            logging.info("=" * 70)

            # Stage 1: Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()

        except Exception as e:
            raise USvisaException(e, sys) from e

