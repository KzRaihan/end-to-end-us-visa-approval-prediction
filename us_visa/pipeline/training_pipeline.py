# ==============================================================================
# us_visa/pipeline/training_pipeline.py
#
# Master Orchestrator:
# Executes all ML pipeline components in the correct sequence.
#
# Current Pipeline:
#     1. Data Ingestion
#     2. Data Validation
#     3. Data Transformation
#     4. Model Trainer

# Future Pipeline:
#     5. Model Evaluation
#     6. Model Pusher
# ==============================================================================


import sys

from us_visa.exception import USvisaException
from us_visa.logger import logging

# ==============================================================================
# Import Pipeline Components
# ==============================================================================

from us_visa.components.data_ingestion import DataIngestion
from us_visa.components.data_validation import DataValidation
from us_visa.components.data_transformation import DataTransformation
from us_visa.components.model_trainer import ModelTrainer
from us_visa.components.model_evaluation import ModelEvaluation


# ==============================================================================
# Import Configuration Entities
# ==============================================================================

from us_visa.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)


# ==============================================================================
# Import Artifact Entities
# ==============================================================================

from us_visa.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact
)


# ==============================================================================
# TRAINING PIPELINE
# ==============================================================================

class TrainingPipeline:
    """
    Master orchestrator for the US Visa ML Training Pipeline.

    This class controls the execution order of all ML pipeline components.

    Current Pipeline Flow:

        Data Ingestion
              ↓
        Data Validation
              ↓
        Data Transformation

    Each component receives the required configuration and/or artifact
    and returns an artifact that can be consumed by the next component.
    """

    def __init__(self):
        """
        Initialize all pipeline configuration objects.

        Configuration objects contain the paths and settings required
        by their corresponding pipeline components.
        """

        # Configuration for Data Ingestion
        self.data_ingestion_config = DataIngestionConfig()

        # Configuration for Data Validation
        self.data_validation_config = DataValidationConfig()

        # Configuration for Data Transformation
        self.data_transformation_config = DataTransformationConfig()

        # Configuration for Model Trainer
        self. model_trainer_config = ModelTrainerConfig()

        # Configuration for Model Evaluation
        self.model_evaluation_config = ModelEvaluationConfig()


    

    # ==========================================================================
    # STAGE 1: DATA INGESTION
    # ==========================================================================

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Execute the Data Ingestion component.

        Flow:
            DataIngestionConfig
                    ↓
            DataIngestion Component
                    ↓
            DataIngestionArtifact

        Returns:
            DataIngestionArtifact:
                Contains paths and information about the
                ingested training and testing datasets.
        """

        try:

            # ------------------------------------------------------------------
            # Step 1: Log the beginning of Data Ingestion
            # ------------------------------------------------------------------

            logging.info(
                ">>> STAGE 1: DATA INGESTION — Starting..."
            )

            # ------------------------------------------------------------------
            # Step 2: Initialize Data Ingestion Component
            # ------------------------------------------------------------------

            # Pass the Data Ingestion configuration to the component.
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            logging.info(
                "Initialized DataIngestion component successfully."
            )

            # ------------------------------------------------------------------
            # Step 3: Execute Data Ingestion
            # ------------------------------------------------------------------

            # Fetch data from MongoDB and create the DataIngestionArtifact.
            data_ingestion_artifact = (
                data_ingestion.initiate_data_ingestion()
            )

            # ------------------------------------------------------------------
            # Step 4: Log Successful Completion
            # ------------------------------------------------------------------

            logging.info(
                ">>> STAGE 1: DATA INGESTION — Completed successfully."
            )

            logging.info(
                "Training and testing datasets are available."
            )

            # ------------------------------------------------------------------
            # Step 5: Return Artifact
            # ------------------------------------------------------------------

            # This artifact will be passed to the Data Validation stage.
            return data_ingestion_artifact

        except Exception as e:

            # Convert the original error into the project's custom exception.
            raise USvisaException(e, sys) from e


    # ==========================================================================
    # STAGE 2: DATA VALIDATION
    # ==========================================================================

    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """
        Execute the Data Validation component.

        Flow:
            DataIngestionArtifact
                    +
            DataValidationConfig
                    ↓
            DataValidation Component
                    ↓
            DataValidationArtifact

        Args:
            data_ingestion_artifact:
                Artifact generated by Data Ingestion.

        Returns:
            DataValidationArtifact:
                Contains validation status and validation-related
                artifact information.
        """

        try:

            # ------------------------------------------------------------------
            # Step 1: Log the beginning of Data Validation
            # ------------------------------------------------------------------

            logging.info(
                ">>> STAGE 2: DATA VALIDATION — Starting..."
            )

            # ------------------------------------------------------------------
            # Step 2: Initialize Data Validation Component
            # ------------------------------------------------------------------

            # Pass the Data Ingestion artifact and Data Validation
            # configuration to the validation component.
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config
            )

            logging.info(
                "Initialized DataValidation component successfully."
            )

            # ------------------------------------------------------------------
            # Step 3: Execute Data Validation
            # ------------------------------------------------------------------

            # Validate the dataset according to the project schema
            # and validation requirements.
            data_validation_artifact = (
                data_validation.initiate_data_validation()
            )

            # ------------------------------------------------------------------
            # Step 4: Log Successful Completion
            # ------------------------------------------------------------------

            logging.info(
                ">>> STAGE 2: DATA VALIDATION — Completed successfully."
            )

            # ------------------------------------------------------------------
            # Step 5: Return Artifact
            # ------------------------------------------------------------------

            # Pass the validation artifact to the next pipeline stage.
            return data_validation_artifact

        except Exception as e:

            # Convert the original error into the project's custom exception.
            raise USvisaException(e, sys) from e


    # ==========================================================================
    # STAGE 3: DATA TRANSFORMATION
    # ==========================================================================

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        """
        Execute the Data Transformation component.

        Flow:
            DataIngestionArtifact
                    +
            DataValidationArtifact
                    +
            DataTransformationConfig
                    ↓
            DataTransformation Component
                    ↓
            DataTransformationArtifact

        Args:
            data_ingestion_artifact:
                Artifact generated by Data Ingestion.

            data_validation_artifact:
                Artifact generated by Data Validation.

        Returns:
            DataTransformationArtifact:
                Contains paths to the transformed datasets and
                the saved preprocessing object.
        """

        try:

            # ------------------------------------------------------------------
            # Step 1: Log the beginning of Data Transformation
            # ------------------------------------------------------------------

            logging.info(
                ">>> STAGE 3: DATA TRANSFORMATION — Starting..."
            )

            # ------------------------------------------------------------------
            # Step 2: Initialize Data Transformation Component
            # ------------------------------------------------------------------

            # Pass the artifacts from the previous stages and
            # the transformation configuration.
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
                data_validation_artifact=data_validation_artifact
            )

            logging.info(
                "Initialized DataTransformation component successfully."
            )

            # ------------------------------------------------------------------
            # Step 3: Execute Data Transformation
            # ------------------------------------------------------------------

            # Perform:
            #     - Feature engineering
            #     - Feature encoding
            #     - Numerical transformation
            #     - Feature scaling
            #     - Training-data resampling
            #
            # The component returns a DataTransformationArtifact.
            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )

            # ------------------------------------------------------------------
            # Step 4: Log Successful Completion
            # ------------------------------------------------------------------

            logging.info(
                ">>> STAGE 3: DATA TRANSFORMATION — Completed successfully."
            )

            # ------------------------------------------------------------------
            # Step 5: Return Artifact
            # ------------------------------------------------------------------

            # This artifact will be passed to the Model Trainer stage.
            return data_transformation_artifact

        except Exception as e:

            # Convert the original error into the project's custom exception.
            raise USvisaException(e, sys) from e

    # ==============================================================================
    # STAGE 4: MODEL TRAINING
    # ==============================================================================

    def start_model_trainer(
        self,
        data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        """
        Execute the Model Trainer component.

        This method receives the artifact generated by the Data Transformation
        stage and passes it to the Model Trainer component.

        Flow:
            DataTransformationArtifact
                    ↓
            ModelTrainerConfig
                    ↓
            ModelTrainer Component
                    ↓
            ModelTrainerArtifact

        Args:
            data_transformation_artifact:
                Artifact containing paths to the transformed training/testing
                datasets and the fitted preprocessing object.

        Returns:
            ModelTrainerArtifact:
                Contains the path to the trained model and model evaluation
                metrics.
        """

        try:

            # ----------------------------------------------------------------------
            # Step 1: Log the beginning of Model Training
            # ----------------------------------------------------------------------

            logging.info(
                ">>> STAGE 4: MODEL TRAINING — Starting..."
            )


            # ----------------------------------------------------------------------
            # Step 2: Initialize Model Trainer Component
            # ----------------------------------------------------------------------

            # Pass:
            #     - Data Transformation artifact
            #     - Model Trainer configuration
            #
            # to the ModelTrainer component.
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )


            # ----------------------------------------------------------------------
            # Step 3: Execute Model Training
            # ----------------------------------------------------------------------

            # The ModelTrainer component:
            #     - Loads transformed train/test data
            #     - Trains candidate models
            #     - Performs hyperparameter tuning
            #     - Selects the best model
            #     - Evaluates the model
            #     - Saves the final model
            model_trainer_artifact = (
                model_trainer.initiate_model_trainer()
            )


            # ----------------------------------------------------------------------
            # Step 4: Log Successful Completion
            # ----------------------------------------------------------------------

            logging.info(
                ">>> STAGE 4: MODEL TRAINING — Completed successfully."
            )


            # ----------------------------------------------------------------------
            # Step 5: Return Model Trainer Artifact
            # ----------------------------------------------------------------------

            # Pass the ModelTrainerArtifact to the next pipeline stage.
            return model_trainer_artifact


        except Exception as e:

            # Convert the original exception into the project's custom
            # USvisaException while preserving the original traceback.



            raise USvisaException(e, sys) from e


    # ==============================================================================
    # STAGE 5: MODEL EVALUATION
    # ==============================================================================

    def start_model_evaluation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ) -> ModelEvaluationArtifact:
        """
        Execute the Model Evaluation component.

        The Model Evaluation component compares the newly trained model
        with the existing production model stored in AWS S3.

        Args:
            data_ingestion_artifact:
                Artifact generated by the Data Ingestion stage.
                Provides the path to the test dataset.

            model_trainer_artifact:
                Artifact generated by the Model Trainer stage.
                Provides the newly trained model path and evaluation metrics.

        Returns:
            ModelEvaluationArtifact:
                Contains the model acceptance status, model paths,
                and performance difference.

        Raises:
            USvisaException:
                If any error occurs during model evaluation.
        """

        try:

            # ----------------------------------------------------------------------
            # Log the beginning of Model Evaluation.
            # ----------------------------------------------------------------------

            logging.info(
                ">>> STAGE 5: MODEL EVALUATION — Starting..."
            )


            # ----------------------------------------------------------------------
            # Initialize the Model Evaluation component.
            #
            # Pass:
            #   1. Model Evaluation configuration
            #   2. Data Ingestion artifact
            #   3. Model Trainer artifact
            # ----------------------------------------------------------------------

            model_evaluation = ModelEvaluation(
                model_eval_config=self.model_evaluation_config,
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact
            )


            # ----------------------------------------------------------------------
            # Execute Model Evaluation.
            #
            # The component compares:
            #
            #   Newly trained model
            #             VS
            #   Existing production model in S3
            # ----------------------------------------------------------------------

            model_evaluation_artifact = (
                model_evaluation.initiate_model_evaluation()
            )


            # ----------------------------------------------------------------------
            # Log the successful completion of Model Evaluation.
            # ----------------------------------------------------------------------

            logging.info(
                ">>> STAGE 5: MODEL EVALUATION — Completed."
            )

            logging.info(
                f"Model Evaluation Artifact: "
                f"{model_evaluation_artifact}"
            )


            # ----------------------------------------------------------------------
            # Return the evaluation artifact.
            # ----------------------------------------------------------------------

            return model_evaluation_artifact


        except Exception as e:

            # Convert the original exception into the project's
            # custom USvisaException with detailed error information.
            raise USvisaException(e, sys) from e



    # ==============================================================================
    # MASTER PIPELINE: EXECUTE ALL ML PIPELINE STAGES
    # ==============================================================================

    def run_pipeline(self) -> None:
        """
        Execute the complete US Visa Machine Learning training pipeline.

        Pipeline Flow:

            Data Ingestion
                ↓
            Data Validation
                ↓
            Data Transformation
                ↓
            Model Training
                ↓
            Model Evaluation

        Each pipeline component generates an artifact that is passed
        to the next component.

        Returns:
            None

        Raises:
            USvisaException:
                If any pipeline stage fails.
        """

        try:

            # ==========================================================================
            # PIPELINE START
            # ==========================================================================

            logging.info("=" * 80)
            logging.info(
                "US VISA ML TRAINING PIPELINE — EXECUTION STARTED"
            )
            logging.info("=" * 80)


            # ==========================================================================
            # STAGE 1: DATA INGESTION
            # ==========================================================================

            # Fetch data from MongoDB and create:
            #   - train.csv
            #   - test.csv
            #
            # Output:
            #   DataIngestionArtifact

            data_ingestion_artifact = self.start_data_ingestion()


            # ==========================================================================
            # STAGE 2: DATA VALIDATION
            # ==========================================================================

            # Validate the ingested datasets against the expected schema
            # and perform data quality/drift checks.
            #
            # Input:
            #   DataIngestionArtifact
            #
            # Output:
            #   DataValidationArtifact

            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )


            # ==========================================================================
            # STAGE 3: DATA TRANSFORMATION
            # ==========================================================================

            # Perform:
            #   - Feature engineering
            #   - Encoding
            #   - Scaling
            #   - Power transformation
            #   - Handling class imbalance
            #
            # Input:
            #   DataIngestionArtifact
            #   DataValidationArtifact
            #
            # Output:
            #   DataTransformationArtifact

            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )


            # ==========================================================================
            # STAGE 4: MODEL TRAINING
            # ==========================================================================

            # Train multiple ML classification models and select
            # the best-performing model based on the configured metric.
            #
            # Input:
            #   DataTransformationArtifact
            #
            # Output:
            #   ModelTrainerArtifact

            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )


            # ==========================================================================
            # STAGE 5: MODEL EVALUATION
            # ==========================================================================

            # Compare the newly trained model with the existing
            # production model stored in AWS S3.
            #
            # Input:
            #   DataIngestionArtifact
            #   ModelTrainerArtifact
            #
            # Output:
            #   ModelEvaluationArtifact

            model_evaluation_artifact = self.start_model_evaluation(
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact
            )


            # ==========================================================================
            # PIPELINE COMPLETED
            # ==========================================================================

            logging.info("=" * 80)
            logging.info(
                "US VISA ML TRAINING PIPELINE — EXECUTION COMPLETED"
            )
            logging.info("=" * 80)

            logging.info(
                f"Final Model Evaluation Artifact: "
                f"{model_evaluation_artifact}"
            )


        except Exception as e:

            # Convert any pipeline error into the project's
            # custom exception for consistent error handling.
            raise USvisaException(e, sys) from e










    