# ==============================================================================
# us_visa/components/model_evaluation.py
#
# Model Evaluation Component
#
# Purpose:
#     Compare the newly trained US Visa model with the existing model
#     registered in AWS S3 and determine whether the new model should
#     be accepted.
#
# Evaluation Flow:
#
#     Model Trainer
#          │
#          ▼
#     Newly Trained Model
#          │
#          │
#          ├───────────────┐
#          │               │
#          ▼               ▼
#     New Model F1     Existing Model F1
#                          │
#                      AWS S3
#                          │
#          └───────┬───────┘
#                  ▼
#           Compare F1 Scores
#                  │
#          ┌───────┴───────┐
#          ▼               ▼
#       New Model       Existing Model
#        Better          Better/Equal
#          │               │
#          ▼               ▼
#       ACCEPT            REJECT
# ==============================================================================


import sys
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.metrics import f1_score

from us_visa.constants import CURRENT_YEAR, TARGET_COLUMN
from us_visa.entity.artifact_entity import (
    DataIngestionArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact
)
from us_visa.entity.estimator import TargetValueMapping
from us_visa.entity.s3_estimator import USvisaEstimator
from us_visa.entity.config_entity import ModelEvaluationConfig
from us_visa.exception import USvisaException
from us_visa.logger import logging


# ==============================================================================
# MODEL EVALUATION RESPONSE
# ==============================================================================

@dataclass
class EvaluateModelResponse:
    """
    Stores the result of comparing the newly trained model with
    the existing production model.

    Attributes:
        trained_model_f1_score:
            F1-score of the newly trained model.

        best_model_f1_score:
            F1-score of the existing model stored in S3.

        is_model_accepted:
            Indicates whether the newly trained model performs better.

        difference:
            Difference between the new model F1-score and existing
            model F1-score.
    """

    # F1-score of the newly trained model.
    trained_model_f1_score: float

    # F1-score of the existing/production model.
    best_model_f1_score: Optional[float]

    # True if the newly trained model is accepted.
    is_model_accepted: bool

    # Improvement in F1-score.
    difference: float


# ==============================================================================
# MODEL EVALUATION
# ==============================================================================

class ModelEvaluation:
    """
    Compare the newly trained model against the existing model
    stored in AWS S3.

    The newly trained model is accepted only when its performance
    is better than the existing model.
    """

    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ):
        """
        Initialize the Model Evaluation component.

        Args:
            model_eval_config:
                Configuration required for model evaluation.

            data_ingestion_artifact:
                Artifact containing paths to the original train/test data.

            model_trainer_artifact:
                Artifact containing the newly trained model information
                and its evaluation metrics.
        """

        try:

            # Store Model Evaluation configuration.
            self.model_eval_config = model_eval_config

            # Store Data Ingestion artifact.
            self.data_ingestion_artifact = data_ingestion_artifact

            # Store Model Trainer artifact.
            self.model_trainer_artifact = model_trainer_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e


    # ==========================================================================
    # GET EXISTING / PRODUCTION MODEL
    # ==========================================================================

    def get_best_model(self) -> Optional[USvisaEstimator]:
        """
        Retrieve the existing model registered in AWS S3.

        Returns:
            USvisaEstimator:
                S3 estimator if an existing model is available.

            None:
                If no existing model is found in S3.
        """

        try:

            # Get the S3 bucket name from the configuration.
            bucket_name = self.model_eval_config.bucket_name

            # Get the S3 model key/path from the configuration.
            model_path = self.model_eval_config.s3_model_key_path

            # Create an estimator connected to the model in S3.
            usvisa_estimator = USvisaEstimator(
                bucket_name=bucket_name,
                model_path=model_path
            )

            # Check whether the model exists in S3.
            if usvisa_estimator.is_model_present(
                model_path=model_path
            ):
                logging.info(
                    "Existing production model found in AWS S3."
                )

                return usvisa_estimator

            # No existing model was found.
            logging.info(
                "No existing production model found in AWS S3."
            )

            return None

        except Exception as e:
            raise USvisaException(e, sys) from e


    # ==========================================================================
    # EVALUATE MODELS
    # ==========================================================================

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Compare the newly trained model with the existing production model.

        The comparison is performed using F1-score.

        Returns:
            EvaluateModelResponse:
                Contains both model scores, acceptance status, and
                performance difference.
        """

        try:

            logging.info(
                "Starting model evaluation."
            )

            # ------------------------------------------------------------------
            # Load the original test dataset.
            # ------------------------------------------------------------------

            test_df = pd.read_csv(
                self.data_ingestion_artifact.test_file_path
            )

            logging.info(
                "Loaded test dataset for model evaluation."
            )


            # ------------------------------------------------------------------
            # Create the company_age feature.
            # ------------------------------------------------------------------

            test_df["company_age"] = (
                CURRENT_YEAR - test_df["yr_of_estab"]
            )

            logging.info(
                "Created company_age feature in test dataset."
            )


            # ------------------------------------------------------------------
            # Separate input features and target.
            # ------------------------------------------------------------------

            x = test_df.drop(
                TARGET_COLUMN,
                axis=1
            )

            y = test_df[TARGET_COLUMN]


            # ------------------------------------------------------------------
            # Convert target labels into numerical values.
            #
            # Certified → 0
            # Denied    → 1
            # ------------------------------------------------------------------

            y = y.replace(
                TargetValueMapping()._asdict()
            )


            # ------------------------------------------------------------------
            # Get F1-score of the newly trained model.
            # ------------------------------------------------------------------

            trained_model_f1_score = (
                self.model_trainer_artifact
                .metric_artifact
                .f1_score
            )

            logging.info(
                f"Newly trained model F1-score: "
                f"{trained_model_f1_score:.4f}"
            )


            # ------------------------------------------------------------------
            # Get the existing production model from S3.
            # ------------------------------------------------------------------

            best_model_f1_score = None

            best_model = self.get_best_model()


            # ------------------------------------------------------------------
            # Evaluate the existing production model.
            # ------------------------------------------------------------------

            if best_model is not None:

                # Generate predictions using the existing model.
                y_hat_best_model = best_model.predict(x)

                # Calculate F1-score of the existing model.
                best_model_f1_score = f1_score(
                    y,
                    y_hat_best_model
                )

                logging.info(
                    f"Existing production model F1-score: "
                    f"{best_model_f1_score:.4f}"
                )


            # ------------------------------------------------------------------
            # Handle the case where no production model exists.
            # ------------------------------------------------------------------

            # If no existing model is available, consider its score as zero.
            previous_model_score = (
                0.0
                if best_model_f1_score is None
                else best_model_f1_score
            )


            # ------------------------------------------------------------------
            # Calculate performance difference.
            # ------------------------------------------------------------------

            difference = (
                trained_model_f1_score
                - previous_model_score
            )


            # ------------------------------------------------------------------
            # Decide whether the new model should be accepted.
            # ------------------------------------------------------------------

            is_model_accepted = (
                trained_model_f1_score > previous_model_score
            )


            # ------------------------------------------------------------------
            # Create evaluation response.
            # ------------------------------------------------------------------

            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=is_model_accepted,
                difference=difference
            )


            logging.info(
                f"Model evaluation result: {result}"
            )

            return result

        except Exception as e:
            raise USvisaException(e, sys) from e


    # ==========================================================================
    # INITIATE MODEL EVALUATION
    # ==========================================================================

    def initiate_model_evaluation(
        self
    ) -> ModelEvaluationArtifact:
        """
        Execute the complete Model Evaluation process.

        Returns:
            ModelEvaluationArtifact:
                Artifact containing the model acceptance status,
                performance difference, and model paths.
        """

        try:

            logging.info(
                "Entered initiate_model_evaluation method "
                "of ModelEvaluation class."
            )


            # ------------------------------------------------------------------
            # Execute model comparison.
            # ------------------------------------------------------------------

            evaluate_model_response = self.evaluate_model()


            # ------------------------------------------------------------------
            # Get the S3 path of the existing model.
            # ------------------------------------------------------------------

            s3_model_path = (
                self.model_eval_config.s3_model_key_path
            )


            # ------------------------------------------------------------------
            # Create Model Evaluation Artifact.
            # ------------------------------------------------------------------

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=(
                    evaluate_model_response.is_model_accepted
                ),

                s3_model_path=s3_model_path,

                trained_model_path=(
                    self.model_trainer_artifact
                    .trained_model_file_path
                ),

                # NOTE:
                # This field represents the change in F1-score,
                # not accuracy.
                changed_accuracy=(
                    evaluate_model_response.difference
                )
            )


            logging.info(
                f"Model evaluation artifact: "
                f"{model_evaluation_artifact}"
            )

            logging.info(
                "Exited initiate_model_evaluation method "
                "of ModelEvaluation class."
            )

            return model_evaluation_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e