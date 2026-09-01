# ==============================================================================
# us_visa/cloud_storage/s3_estimator.py
#
# S3 Model Estimator
#
# Purpose:
#     Provides a high-level interface for storing, loading, checking, and
#     predicting with the US Visa model stored in AWS S3.
#
# Responsibilities:
#     1. Connect to the configured S3 bucket.
#     2. Check whether a model exists in S3.
#     3. Load a trained model from S3.
#     4. Upload a trained model to S3.
#     5. Cache the loaded model for prediction.
# ==============================================================================


import sys

from pandas import DataFrame

from us_visa.cloud_storage.aws_storage import SimpleStorageService
from us_visa.entity.estimator import USvisaModel
from us_visa.exception import USvisaException


# ==============================================================================
# US VISA ESTIMATOR
# ==============================================================================

class USvisaEstimator:
    """
    Provides an interface for managing the US Visa ML model in AWS S3.

    This class uses SimpleStorageService to communicate with S3 and
    USvisaModel to perform predictions.

    Attributes:
        bucket_name:
            Name of the AWS S3 bucket containing the model.

        model_path:
            S3 key/path of the model.

        s3:
            SimpleStorageService instance used for S3 operations.

        loaded_model:
            Cached USvisaModel object loaded from S3.
    """

    def __init__(
        self,
        bucket_name: str,
        model_path: str
    ):
        """
        Initialize the US Visa S3 estimator.

        Args:
            bucket_name:
                Name of the S3 bucket where the model is stored.

            model_path:
                S3 key/path of the model file.
        """

        # Store the S3 bucket name.
        self.bucket_name = bucket_name

        # Create the S3 storage service.
        self.s3 = SimpleStorageService()

        # Store the S3 model path/key.
        self.model_path = model_path

        # The model is loaded lazily when predict() is called.
        self.loaded_model: USvisaModel | None = None


    # ==========================================================================
    # CHECK MODEL EXISTENCE
    # ==========================================================================

    def is_model_present(self, model_path: str) -> bool:
        """
        Check whether a model exists at the specified S3 path.

        Args:
            model_path:
                S3 object key/path of the model.

        Returns:
            bool:
                True if the model exists in S3, otherwise False.
        """

        try:

            logging_message = (
                f"Checking whether model exists at S3 path: {model_path}"
            )

            # Check whether the specified S3 object exists.
            return self.s3.s3_key_path_available(
                bucket_name=self.bucket_name,
                s3_key=model_path
            )

        except USvisaException as e:

            # If the S3 check fails, log/print the exception and
            # return False so the caller knows the model is unavailable.
            print(e)
            return False


    # ==========================================================================
    # LOAD MODEL
    # ==========================================================================

    def load_model(self) -> USvisaModel:
        """
        Load the trained US Visa model from AWS S3.

        Returns:
            USvisaModel:
                The trained model loaded from S3.
        """

        try:

            # Retrieve and deserialize the model from S3.
            return self.s3.load_model(
                self.model_path,
                bucket_name=self.bucket_name
            )

        except Exception as e:

            raise USvisaException(e, sys) from e


    # ==========================================================================
    # SAVE MODEL
    # ==========================================================================

    def save_model(
        self,
        from_file: str,
        remove: bool = False
    ) -> None:
        """
        Upload a local trained model to AWS S3.

        Args:
            from_file:
                Local path of the trained model.

            remove:
                If True, remove the local model after successful upload.
                Default is False.

        Returns:
            None
        """

        try:

            # Upload the local model to the configured S3 location.
            self.s3.upload_file(
                from_file=from_file,
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove
            )

        except Exception as e:

            raise USvisaException(e, sys) from e


    # ==========================================================================
    # MODEL PREDICTION
    # ==========================================================================

    def predict(self, dataframe: DataFrame):
        """
        Generate predictions using the model stored in S3.

        The model is loaded only once and then cached in memory.
        This avoids downloading the model from S3 for every prediction.

        Args:
            dataframe:
                Input features used for prediction.

        Returns:
            Model predictions.
        """

        try:

            # Load the model only if it has not already been loaded.
            if self.loaded_model is None:
                self.loaded_model = self.load_model()

            # Use the loaded model to generate predictions.
            return self.loaded_model.predict(
                dataframe=dataframe
            )

        except Exception as e:

            raise USvisaException(e, sys) from e