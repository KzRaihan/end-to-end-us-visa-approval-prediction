# ==============================================================================
# us_visa/components/data_ingestion.py
# Component 1: Data Ingestion — Fetch, Split, Save
# ==============================================================================

import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from us_visa.entity.config_entity import DataIngestionConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.data_access.usvisa_data import USvisaData


class DataIngestion:
    """
    First pipeline component responsible for:
    1. Extracting raw data from MongoDB
    2. Performing stratified train-test split
    3. Saving train.csv and test.csv to the artifacts directory
    4. Returning a DataIngestionArtifact with output file paths
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        Args:
            data_ingestion_config: Configuration dataclass containing
            database name, collection name, split ratio, and output paths.
        """
        try:
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise USvisaException(e, sys)

    # ------------------------------------------------------------------
    # STEP 1: Fetch raw data from MongoDB
    # ------------------------------------------------------------------
    def export_data_into_feature_store(self) -> DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from mongodb to csv file
        
        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info(f"Exporting data from mongodb")
            # Initialize data access object (uses singleton MongoDB connection)
            usvisa_data = USvisaData()

            # Fetch all records as DataFrame (handles _id removal internally)
            dataframe = usvisa_data.export_collection_as_dataframe(collection_name=
                                                                   self.data_ingestion_config.collection_name)
            logging.info(f"Shape of dataframe: {dataframe.shape}")

            # Save raw snapshot to feature_store for audit/reproducibility
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")

            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe

            
        except Exception as e:
            raise USvisaException(e, sys)

    # ------------------------------------------------------------------
    # STEP 2: Split data into train and test sets
    # ------------------------------------------------------------------
    def split_data_as_train_test(self,dataframe: DataFrame) -> None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio 
        
        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            # Stratified split preserves class distribution in both sets
            train_set, test_set = train_test_split(
                dataframe, 
                test_size=self.data_ingestion_config.train_test_split_ratio
                )

            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting train and test file path.")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header = True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index = False, header = True)

            logging.info(f"Exported train and test file path.")


        except Exception as e:
            raise USvisaException(e, sys) from e

    
    # ------------------------------------------------------------------
    # STEP 3: Orchestrate full ingestion and return artifact
    # ------------------------------------------------------------------
    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        
        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            # Step 1: Fetch from MongoDB and save raw snapshot
            dataframe = self.export_data_into_feature_store()
            logging.info("Got the data from mongodb")

            # Step 2: Stratified train-test split
            self.split_data_as_train_test(dataframe)
            logging.info("Performed train test split on the dataframe")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            # Step 5: Build and return the output artifact
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
    
            logging.info("DATA INGESTION COMPLETED SUCCESSFULLY")

            return data_ingestion_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e

