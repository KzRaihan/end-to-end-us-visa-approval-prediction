# ==============================================================================
# us_visa/data_access/usvisa_data.py
# Data Access Data from MongoDB and convert data into DataFrame
# ==============================================================================
import os
import sys
import pandas as pd
import numpy as np
from typing import Optional


from us_visa.configuration.mongo_db_connection import MongoDBClient
from us_visa.constants import DATABASE_NAME
from us_visa.exception import USvisaException


class USvisaData:
    """
    This class help to export entire mongo db record as pandas dataframe 
    
    """
    def __init__(self):
        """  
        Initializes the data access layer (connection client) with a MongoDB connection.
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)

        except Exception as e:
            raise USvisaException(e, sys)

    def export_collection_as_dataframe(self, collection_name:str, database_name: Optional[str]=None) -> pd.DataFrame:
        """
        Reads ALL documents from a MongoDB collection and converts to a Pandas DataFrame.

        Args:
            collection_name: Name of the collection to read from.
            database_name: Optional override for the database name.

        Returns:
            pd.DataFrame containing all records from the collection.
        """
        try:
             # Step 1: Access the target collection
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]


            # Step 2: Fetch all documents as a list of dictionaries and convert to dataframe
            df = pd.DataFrame(list(collection.find()))

            # Step 3: Remove MongoDB internal '_id' field (not useful for ML)
            if "_id" in df.columns.tolist():
                
                df = df.drop(columns=["_id"], axis=1)
            # replace na value with np.nan
            df.replace({"na": np.nan}, inplace=True)

            return df


        except Exception as e:
            raise USvisaException(e,sys)


