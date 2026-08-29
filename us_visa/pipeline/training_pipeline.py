# ==============================================================================
# us_visa/pipeline/training_pipeline.py
# Master Orchestrator: Executes all ML pipeline components in sequence
# ==============================================================================
import sys
from us_visa.exception import USvisaException
from us_visa.logger import logging

from us_visa.components.data_ingestion import DataIngestion
from us_visa.components.data_validation import DataValidation

from us_visa.entity.config_entity import(DataIngestionConfig,
                                         DataValidationConfig
)

from us_visa.entity.artifact_entity import(DataIngestionArtifact, 
                                           DataValidationArtifact
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
        self.data_validation_config = DataValidationConfig()
        


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
    # STAGE 2: DATA VALIDATION
    # ------------------------------------------------------------------
    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """
        This method of TrainPipeline class is responsible for
        starting the Data Validation component.

        Flow:
            DataIngestionArtifact
                    +
            DataValidationConfig
                    ↓
            DataValidation Component
                    ↓
            DataValidationArtifact
        """

        try:

            # ------------------------------------------------------------------
            # STEP 1: Log the start of the Data Validation stage
            # ------------------------------------------------------------------
            logging.info(
                ">>> STAGE 2: DATA VALIDATION — Starting..."
            )


            # ------------------------------------------------------------------
            # STEP 2: Create DataValidation Component
            # ------------------------------------------------------------------
            # Pass the output artifact from the Data Ingestion component
            # and the configuration required by the Data Validation component.
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config
            )


            # ------------------------------------------------------------------
            # STEP 3: Initiate Data Validation
            # ------------------------------------------------------------------
            # The DataValidation component performs operations such as:
            #   - Validating dataset schema
            #   - Detecting data drift
            #   - Generating validation artifacts
            data_validation_artifact = (
                data_validation.initiate_data_validation()
            )


            # ------------------------------------------------------------------
            # STEP 4: Log Successful Data Validation
            # ------------------------------------------------------------------
            logging.info(
                "Performed the data validation operation successfully."
            )


            # ------------------------------------------------------------------
            # STEP 5: Log the Exit from This Method
            # ------------------------------------------------------------------
            logging.info(
                "Exited the start_data_validation method "
                "of TrainPipeline class."
            )


            # ------------------------------------------------------------------
            # STEP 6: Return Data Validation Artifact
            # ------------------------------------------------------------------
            # Return the artifact generated by the Data Validation component
            # so that the next pipeline stage can use it.
            return data_validation_artifact


        # ------------------------------------------------------------------
        # EXCEPTION HANDLING
        # ------------------------------------------------------------------
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

            # Stage 2: Data Validation
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact = data_ingestion_artifact
            )


        except Exception as e:
            raise USvisaException(e, sys) from e

