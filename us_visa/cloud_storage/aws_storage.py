# ==============================================================================
# us_visa/cloud_storage/aws_storage.py
#
# AWS S3 Storage Module
#
# Purpose:
#     Provides utility methods for storing and retrieving files from
#     an AWS S3 bucket.
#
# Responsibilities:
#     1. Use the centralized AWS S3 connection.
#     2. Upload files to S3.
#     3. Download files from S3.
#     4. Check whether objects exist in S3.
#     5. Provide a reusable interface for Model Evaluation and Model Pusher.
#
# Architecture:
#
#     AWS Credentials
#            ↓
#     configuration/aws_connection.py
#            ↓
#          S3Client
#            ↓
#     cloud_storage/aws_storage.py
#            ↓
#     S3 Bucket / Model Registry
# ==============================================================================


import os
import sys

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.configuration.aws_connection import S3Client


# ==============================================================================
# AWS S3 STORAGE
# ==============================================================================

class SimpleStorageService:
    """
    Provides high-level operations for interacting with AWS S3.

    This class uses the centralized S3Client from the configuration layer
    and provides reusable methods for uploading and downloading files.

    Attributes:
        s3_client:
            Boto3 S3 client used to communicate with AWS S3.

        s3_resource:
            Boto3 S3 resource used for higher-level S3 operations.
    """

    def __init__(self):
        """
        Initialize the S3 storage service.

        The AWS connection is obtained from the centralized S3Client.
        """

        # Create/reuse the centralized AWS S3 connection.
        s3_connection = S3Client()

        # Store the boto3 S3 client.
        self.s3_client = s3_connection.s3_client

        # Store the boto3 S3 resource.
        self.s3_resource = s3_connection.s3_resource


    # ==========================================================================
    # UPLOAD FILE
    # ==========================================================================

    def upload_file(
        self,
        from_filename: str,
        to_filename: str,
        bucket_name: str
    ) -> None:
        """
        Upload a local file to an AWS S3 bucket.

        Args:
            from_filename:
                Local path of the file to upload.

            to_filename:
                Destination key/path inside the S3 bucket.

            bucket_name:
                Name of the target S3 bucket.

        Returns:
            None
        """

        try:

            logging.info(
                f"Uploading file '{from_filename}' "
                f"to S3 bucket '{bucket_name}'."
            )

            # Upload the local file to the specified S3 bucket.
            self.s3_client.upload_file(
                Filename=from_filename,
                Bucket=bucket_name,
                Key=to_filename
            )

            logging.info(
                f"Successfully uploaded '{from_filename}' "
                f"to S3 as '{to_filename}'."
            )

        except Exception as e:

            # Convert the original exception into the project's
            # custom exception while preserving the traceback.
            raise USvisaException(e, sys) from e


    # ==========================================================================
    # DOWNLOAD FILE
    # ==========================================================================

    def download_file(
        self,
        from_filename: str,
        to_filename: str,
        bucket_name: str
    ) -> None:
        """
        Download a file from AWS S3 to the local machine.

        Args:
            from_filename:
                S3 object key/path.

            to_filename:
                Local path where the downloaded file will be saved.

            bucket_name:
                Name of the S3 bucket.

        Returns:
            None
        """

        try:

            logging.info(
                f"Downloading '{from_filename}' "
                f"from S3 bucket '{bucket_name}'."
            )

            # Download the S3 object to the local destination.
            self.s3_client.download_file(
                Bucket=bucket_name,
                Key=from_filename,
                Filename=to_filename
            )

            logging.info(
                f"Successfully downloaded '{from_filename}' "
                f"to '{to_filename}'."
            )

        except Exception as e:

            raise USvisaException(e, sys) from e