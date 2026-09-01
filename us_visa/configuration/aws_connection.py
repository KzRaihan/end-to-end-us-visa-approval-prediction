# ==============================================================================
# us_visa/configuration/aws_connection.py
#
# AWS Connection Module
#
# Purpose:
#     Creates and manages connections to AWS S3 using boto3.
#
# Responsibilities:
#     1. Read AWS credentials from environment variables.
#     2. Validate that required credentials are available.
#     3. Create an S3 resource connection.
#     4. Create an S3 client connection.
#     5. Reuse the existing connections instead of creating them repeatedly.
#
# Used by:
#     - Cloud Storage
#     - Model Evaluation
#     - Model Pusher
# ==============================================================================


import os

import boto3

from us_visa.constants import (
    AWS_ACCESS_KEY_ID_ENV_KEY,
    AWS_SECRET_ACCESS_KEY_ENV_KEY,
    REGION_NAME
)


# ==============================================================================
# S3 CLIENT
# ==============================================================================

class S3Client:
    """
    Manage AWS S3 client and resource connections.

    The class uses AWS credentials stored in environment variables
    and creates reusable boto3 S3 connections.

    Class Attributes:
        s3_client:
            Boto3 S3 client object.

        s3_resource:
            Boto3 S3 resource object.
    """

    # --------------------------------------------------------------------------
    # Class-level connection objects
    # --------------------------------------------------------------------------
    # Initially set to None.
    #
    # These are shared across instances so that multiple S3Client objects
    # do not unnecessarily create multiple AWS connections.
    s3_client = None
    s3_resource = None


    # ==========================================================================
    # INITIALIZATION
    # ==========================================================================

    def __init__(self, region_name: str = REGION_NAME):
        """
        Initialize the AWS S3 connection.

        AWS credentials are retrieved from environment variables.

        Args:
            region_name:
                AWS region where the S3 bucket is located.

        Raises:
            Exception:
                If AWS access key or secret access key is not configured.
        """

        # ----------------------------------------------------------------------
        # Create the S3 connections only if they do not already exist.
        # ----------------------------------------------------------------------

        if (
            S3Client.s3_resource is None
            or S3Client.s3_client is None
        ):

            # ------------------------------------------------------------------
            # Read AWS credentials from environment variables.
            # ------------------------------------------------------------------

            access_key_id = os.getenv(
                AWS_ACCESS_KEY_ID_ENV_KEY
            )

            secret_access_key = os.getenv(
                AWS_SECRET_ACCESS_KEY_ENV_KEY
            )


            # ------------------------------------------------------------------
            # Validate AWS Access Key
            # ------------------------------------------------------------------

            if access_key_id is None:
                raise Exception(
                    f"Environment variable "
                    f"'{AWS_ACCESS_KEY_ID_ENV_KEY}' is not set."
                )


            # ------------------------------------------------------------------
            # Validate AWS Secret Access Key
            # ------------------------------------------------------------------

            if secret_access_key is None:
                raise Exception(
                    f"Environment variable "
                    f"'{AWS_SECRET_ACCESS_KEY_ENV_KEY}' is not set."
                )


            # ------------------------------------------------------------------
            # Create Boto3 S3 Resource
            # ------------------------------------------------------------------

            # The resource interface provides a higher-level API for
            # interacting with S3 buckets and objects.
            S3Client.s3_resource = boto3.resource(
                "s3",
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region_name
            )


            # ------------------------------------------------------------------
            # Create Boto3 S3 Client
            # ------------------------------------------------------------------

            # The client interface provides a lower-level API for
            # performing S3 operations.
            S3Client.s3_client = boto3.client(
                "s3",
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region_name
            )


        # ----------------------------------------------------------------------
        # Assign the shared connections to the current instance.
        # ----------------------------------------------------------------------

        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client