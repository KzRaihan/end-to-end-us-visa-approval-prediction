# ==============================================================================
# us_visa/configuration/mongo_db_connection.py
# Singleton MongoDB connection manager for the US Visa ML Pipeline
# ==============================================================================
import os
import sys
import pymongo
import certifi

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.constants import DATABASE_NAME, MONGODB_URL_KEY


ca = certifi.where()

class MongoDBClient:
    """
    Class Name  : MongoDBClient
    Description : Initializes the MongoDB connection.  

                 
    output      : connection to mongodb database and return connection object
    on Failure  : raises an exception
    """
    # Class-level attribute ensures only ONE client instance exists
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        """ Args:
                    database_name: Name of the MongoDB database to connect to. 
        """
        try:
             # Step 1: Check if a connection already exists (Singleton pattern)
            if MongoDBClient.client is None:

                # Step 2: Read MongoDB URL from environment variable
                mongo_db_url = os.getenv(MONGODB_URL_KEY)

                # Step 3: Validate that the URL exists
                if mongo_db_url is None:
                    raise Exception(
                        f"Environment variable '{MONGODB_URL_KEY}' is not set. "
                        f"Please create a .env file with your MongoDB connection string."
                    )

                # Step 4: Establish connection
                # certifi provides SSL certificates for Atlas cloud connections
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

            # Step 5: Store the client and database reference
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB connection successful")

        except Exception as e:
            raise USvisaException(e,sys)
