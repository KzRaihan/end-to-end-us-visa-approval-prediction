import json
import sys

import pandas as pd
from pandas import DataFrame

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from us_visa.exception import USvisaException
from us_visa.logger import logging

from us_visa.constants import SCHEMA_FILE_PATH
from us_visa.utils.main_utils import read_yaml_file, write_yaml_file

from us_visa.entity.config_entity import DataValidationConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact


class DataValidation:   
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        Args:
            data_validation_config: Paths for report output.
            data_ingestion_artifact: Paths to train.csv and test.csv from Stage 1.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            logging.info("DataValidation component initialized.")

        except Exception as e:
            raise USvisaException(e, sys) from e

    # ------------------------------------------------------------------
    # CHECK 1: Validate column names
    # ------------------------------------------------------------------

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column present: [{status}]")
            return status

        except Exception as e:
            raise USvisaException(e, sys) from e

    # ------------------------------------------------------------------
    # CHECK 2: Validate numerical and Categorical columns
    # ------------------------------------------------------------------
    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Method Name :   is_column_exist
        Description :   This method validates the existence of a numerical and categorical columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []

            # Check Numerical columns
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")

            # check Categorical columns
            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns)>0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True

        except Exception as e:
            raise USvisaException(e, sys) from e


    # ------------------------------------------------------------------
    # HELPER: Load DataFrame
    # ------------------------------------------------------------------
    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys) from e

    # ------------------------------------------------------------------
    # METHOD: DETECT DATASET DRIFT
    # ------------------------------------------------------------------
    def detect_dataset_drift(
        self,
        reference_df: DataFrame,
        current_df: DataFrame,
    ) -> bool:
        """
        Method Name : detect_dataset_drift

        Description :
            This method detects whether dataset drift is present
            between the reference and current datasets.

        Output :
            Returns True if dataset drift is detected,
            otherwise returns False.

        On Failure :
            Raises USvisaException.
        """

        try:

            # ------------------------------------------------------------------
            # STEP 1: Create Evidently Data Drift Report
            # ------------------------------------------------------------------
            # DataDriftPreset() contains the metrics required to
            # evaluate data drift across the dataset.
            data_drift_report = Report(
                metrics=[
                    DataDriftPreset()
                ]
            )


            # ------------------------------------------------------------------
            # STEP 2: Run Data Drift Analysis
            # ------------------------------------------------------------------
            # reference_df = baseline/training data
            # current_df   = new/test data
            data_drift_report.run(
                reference_data=reference_df,
                current_data=current_df
            )


            # ------------------------------------------------------------------
            # STEP 3: Convert Evidently Report to Dictionary
            # ------------------------------------------------------------------
            # This allows us to access the calculated drift results
            # programmatically.
            json_report = data_drift_report.as_dict()


            # ------------------------------------------------------------------
            # STEP 4: Extract Dataset-Level Drift Result
            # ------------------------------------------------------------------
            # DataDriftPreset() generates DatasetDriftMetric as
            # the first metric in the report.
            dataset_drift_result = json_report["metrics"][0]["result"]


            # ------------------------------------------------------------------
            # STEP 5: Extract Drift Statistics
            # ------------------------------------------------------------------
            n_features = dataset_drift_result["number_of_columns"]

            n_drifted_features = (
                dataset_drift_result["number_of_drifted_columns"]
            )

            drift_status = dataset_drift_result["dataset_drift"]


            # ------------------------------------------------------------------
            # STEP 6: Log Data Drift Information
            # ------------------------------------------------------------------
            logging.info(
                f"{n_drifted_features}/{n_features} "
                f"features drift detected."
            )


            # ------------------------------------------------------------------
            # STEP 7: Save Data Drift Report
            # ------------------------------------------------------------------
            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=json_report
            )


            # ------------------------------------------------------------------
            # STEP 8: Return Dataset Drift Status
            # ------------------------------------------------------------------
            # True  -> Dataset drift detected
            # False -> No dataset drift detected
            return drift_status


        # ------------------------------------------------------------------
        # EXCEPTION HANDLING
        # ------------------------------------------------------------------
        except Exception as e:
            raise USvisaException(e, sys) from e

    # ------------------------------------------------------------------
    # MAIN: Orchestrate all validation checks
    # ------------------------------------------------------------------
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline
                    :   Main entry point. Runs all validation checks and returns artifact.
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            validation_error_msg = ""
            logging.info("=" * 60)
            logging.info("STARTING DATA VALIDATION PIPELINE")
            logging.info("=" * 60)

            # Step 1: Load train and test data from ingestion artifact
            train_df, test_df = (
                DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            )
            # Step 2: checks  Validate column names 
            # ------------------------------------------------------------------

            status = self.validate_number_of_columns(dataframe=test_df)
            logging.info(f"All required columns present in training dataframe: {status}")

            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."


            status = self.validate_number_of_columns(dataframe=test_df)
            logging.info(f"All required columns present in testing dataframe: {status}")


            if not status:
                validation_error_msg += f"Column are missing in test dataframe"

            # Step 3: checks Validate numerical and Categorical of train and test columns
            # --------------------------------------------------------------------------
            
            # for check training set 
            status = self.is_column_exist(df = train_df)

            if not status:
                validation_error_msg += f"columns are Missing in training dataframe."
                
            # for check testing set 
            status = self.is_column_exist(df = test_df)

            if not status:
                validation_error_msg += f"columns are Missing in testing dataframe."

            # Step 4: Run drift detection (train vs test)  
            validation_status = len(validation_error_msg) == 0

            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)

                if drift_status:
                    logging.info(f"Drift Detected.")
                    validation_error_msg = "Drift Detected"
                else:
                    validation_error_msg = "Drift not Detected"
            else:
                logging.info(f"Validation_error: {validation_error_msg}")

            # Step 5: Build and return artifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            logging.info(f"Data Validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        
        except Exception as e:
            raise USvisaException(e, sys) from e
