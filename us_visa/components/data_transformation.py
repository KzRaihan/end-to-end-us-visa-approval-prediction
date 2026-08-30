# ==============================================================================
# us_visa/components/data_transformation.py
# Component 3: Data Transformation
# ==============================================================================
import sys
import numpy as np
import pandas as pd

from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    OrdinalEncoder,
    PowerTransformer
)

from us_visa.utils.main_utils import (
    read_yaml_file,
    drop_columns,
    save_object,
    save_numpy_array_data
)

from us_visa.constants import (
    TARGET_COLUMN,
    SCHEMA_FILE_PATH,
    CURRENT_YEAR
)

from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact

)
from us_visa.entity.estimator import TargetValueMapping

from us_visa.exception import USvisaException
from us_visa.logger import logging

class DataTransformation:
    """
    Third pipeline component responsible for:
    1. Feature engineering (company_age, annual_wage, wage_per_employee)
    2. Target encoding via TargetValueMapping (estimator.py)
    3. Building the sklearn ColumnTransformer preprocessing pipeline
    4. Fitting on train data, transforming train and test data
    5. Saving processed .npy arrays and preprocessor.pkl to artifacts
    """
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """ 
        : param data_ingestion_artifact: output reference of data ingestion artifact stage
        : param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

        except Exception as e:
            raise USvisaException(e, sys) from e

    # -----------------------------------------------------------------
    # STEP 1. read the data
    # -----------------------------------------------------------------
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys) from e

    # ------------------------------------------------------------------
    # STEP 2: Build the ColumnTransformer Preprocessing Pipeline
    # -----------------------------------------------------------------
    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Create and return the preprocessing object for feature transformation.

        The transformer applies different preprocessing techniques to different
        feature groups based on the schema configuration.

        Returns:
            ColumnTransformer: Configured preprocessing object.

        Raises:
            USvisaException: If any error occurs during transformer creation.
        """

        # Log the start of the method for debugging and pipeline monitoring
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )

        try:
            # ------------------------------------------------------------------
            # 1. Initialize individual preprocessing transformers
            # ------------------------------------------------------------------

            # Standardize numerical features:
            # transformed value = (value - mean) / standard deviation
            numerical_transformer = StandardScaler()

            # Convert nominal categorical features into one-hot encoded columns
            oh_transformer = OneHotEncoder()

            # Convert ordinal categorical features into ordered numerical values
            ordinal_encoder = OrdinalEncoder()

            logging.info(
                "Initialized StandardScaler, OneHotEncoder, and OrdinalEncoder"
            )

            # ------------------------------------------------------------------
            # 2. Get feature groups from schema configuration
            # ------------------------------------------------------------------

            # Features that require One-Hot Encoding
            oh_columns = self._schema_config["oh_columns"]

            # Features that have a natural order and require Ordinal Encoding
            or_columns = self._schema_config["or_columns"]

            # Numerical features requiring Power Transformation
            transform_columns = self._schema_config["transform_columns"]

            # Numerical features requiring standard scaling
            num_features = self._schema_config["num_features"]

            logging.info(
                "Retrieved feature groups from schema configuration"
            )

            # ------------------------------------------------------------------
            # 3. Create Power Transformation pipeline
            # ------------------------------------------------------------------

            # Yeo-Johnson transformation helps make skewed numerical
            # distributions more Gaussian-like.
            #
            # Unlike Box-Cox, Yeo-Johnson can handle zero and negative values.
            transform_pipe = Pipeline(
                steps=[
                    (
                        "transformer",
                        PowerTransformer(
                            method="yeo-johnson"
                        )
                    )
                ]
            )

            logging.info(
                "Initialized PowerTransformer with Yeo-Johnson method"
            )

            # ------------------------------------------------------------------
            # 4. Combine all transformations using ColumnTransformer
            # ------------------------------------------------------------------

            # Apply different preprocessing operations to different columns.
            #
            # OneHotEncoder   → nominal categorical features
            # OrdinalEncoder  → ordinal categorical features
            # PowerTransformer → skewed numerical features
            # StandardScaler  → numerical features requiring scaling
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "OneHotEncoder",
                        oh_transformer,
                        oh_columns
                    ),
                    (
                        "OrdinalEncoder",
                        ordinal_encoder,
                        or_columns
                    ),
                    (
                        "Transformer",
                        transform_pipe,
                        transform_columns
                    ),
                    (
                        "StandardScaler",
                        numerical_transformer,
                        num_features
                    )
                ]
            )

            logging.info(
                "Created preprocessing object using ColumnTransformer"
            )

            # ------------------------------------------------------------------
            # 5. Return the complete preprocessing object
            # ------------------------------------------------------------------

            logging.info(
                "Exited get_data_transformer_object method successfully"
            )

            return preprocessor

        except Exception as e:
            # custom exception with detailed system information.
            raise USvisaException(e, sys) from e

    # ------------------------------------------------------------------
    # STEP 3: Main Orchestration Method
    # ------------------------------------------------------------------
    def initiate_data_transformation(self,) -> DataTransformationArtifact:
        """
        Initiate the data transformation component.

        This method:
            1. Verifies data validation status.
            2. Creates the preprocessing object.
            3. Loads validated train and test data.
            4. Separates input features and target.
            5. Creates the company_age feature.
            6. Drops unnecessary columns.
            7. Maps target labels to numerical values.
            8. Fits the preprocessor on training data.
            9. Transforms training and testing features.
            10. Applies SMOTEENN only to the training data.
            11. Saves transformed data and preprocessing object.
            12. Returns DataTransformationArtifact.

        Returns:
            DataTransformationArtifact:
                Contains paths of transformed train/test data and
                the fitted preprocessing object.

        Raises:
            USvisaException:
                If any error occurs during data transformation.
        """

        try:

            # ------------------------------------------------------------------
            # 1. Verify Data Validation
            # ------------------------------------------------------------------

            # Continue transformation only if validation was successful.
            if self.data_validation_artifact.validation_status:

                logging.info("Starting data transformation")

                # ------------------------------------------------------------------
                # 2. Create the preprocessing object
                # ------------------------------------------------------------------

                # Create ColumnTransformer containing all preprocessing steps
                # such as encoding, power transformation, and scaling.
                preprocessor = self.get_data_transformer_object()

                logging.info(
                    "Successfully created the preprocessing object"
                )

                # ------------------------------------------------------------------
                # 3. Load validated train and test datasets
                # ------------------------------------------------------------------

                # Read the training dataset generated by Data Ingestion/
                # Data Validation.
                train_df = DataTransformation.read_data(
                    file_path=self.data_ingestion_artifact.trained_file_path
                )

                # Read the testing dataset.
                test_df = DataTransformation.read_data(
                    file_path=self.data_ingestion_artifact.test_file_path
                )

                logging.info(
                    "Successfully loaded train and test datasets"
                )

                # ------------------------------------------------------------------
                # 4. Separate training features and target
                # ------------------------------------------------------------------

                # Remove the target column from training data.
                input_feature_train_df = train_df.drop(
                    columns=[TARGET_COLUMN],
                    axis=1
                )

                # Extract the target column separately.
                target_feature_train_df = train_df[TARGET_COLUMN]

                logging.info(
                    "Separated training features and target"
                )

                # ------------------------------------------------------------------
                # 5. Create company_age feature
                # ------------------------------------------------------------------

                # Derive company age from the year of establishment.
                input_feature_train_df["company_age"] = (
                    CURRENT_YEAR
                    - input_feature_train_df["yr_of_estab"]
                )

                logging.info(
                    "Created company_age feature for training data"
                )

                # ------------------------------------------------------------------
                # 6. Drop unnecessary training columns
                # ------------------------------------------------------------------

                # Retrieve columns that should not be used for model training.
                drop_cols = self._schema_config["drop_columns"]

                # Remove irrelevant/redundant columns.
                input_feature_train_df = drop_columns(
                    df=input_feature_train_df,
                    cols=drop_cols
                )

                logging.info(
                    "Dropped irrelevant/redundant columns from training data"
                )

                # ------------------------------------------------------------------
                # 7. Encode training target
                # ------------------------------------------------------------------

                # Convert categorical target labels into numerical values.
                #
                # Certified -> 0
                # Denied    -> 1
                target_feature_train_df = target_feature_train_df.map(
                    TargetValueMapping().as_dict()
                )

                logging.info(
                    "Mapped training target labels to numerical values"
                )

                # ------------------------------------------------------------------
                # 8. Separate testing features and target
                # ------------------------------------------------------------------

                # Remove the target column from testing data.
                input_feature_test_df = test_df.drop(
                    columns=[TARGET_COLUMN],
                    axis=1
                )

                # Extract the testing target separately.
                target_feature_test_df = test_df[TARGET_COLUMN]

                logging.info(
                    "Separated testing features and target"
                )

                # ------------------------------------------------------------------
                # 9. Create company_age feature for test data
                # ------------------------------------------------------------------

                # Apply exactly the same feature-engineering logic
                # used for the training dataset.
                input_feature_test_df["company_age"] = (
                    CURRENT_YEAR
                    - input_feature_test_df["yr_of_estab"]
                )

                logging.info(
                    "Created company_age feature for testing data"
                )

                # ------------------------------------------------------------------
                # 10. Drop unnecessary testing columns
                # ------------------------------------------------------------------

                # Apply the same column-dropping logic to test data.
                input_feature_test_df = drop_columns(
                    df=input_feature_test_df,
                    cols=drop_cols
                )

                logging.info(
                    "Dropped irrelevant/redundant columns from testing data"
                )

                # ------------------------------------------------------------------
                # 11. Encode testing target
                # ------------------------------------------------------------------

                # Convert testing target labels using the same mapping
                # used for the training target.
                target_feature_test_df = target_feature_test_df.map(
                    TargetValueMapping().as_dict()
                )

                logging.info(
                    "Mapped testing target labels to numerical values"
                )

                # ------------------------------------------------------------------
                # 12. Fit and transform training features
                # ------------------------------------------------------------------

                # IMPORTANT:
                # The preprocessor learns parameters ONLY from training data.
                input_feature_train_arr = preprocessor.fit_transform(
                    input_feature_train_df
                )

                logging.info(
                    "Fitted and transformed training features"
                )

                # ------------------------------------------------------------------
                # 13. Transform testing features
                # ------------------------------------------------------------------

                # Use the already-fitted preprocessor.
                #
                # IMPORTANT:
                # Do NOT call fit_transform() on test data.
                input_feature_test_arr = preprocessor.transform(
                    input_feature_test_df
                )

                logging.info(
                    "Transformed testing features using fitted preprocessor"
                )

                # ------------------------------------------------------------------
                # 14. Handle class imbalance using SMOTEENN
                # ------------------------------------------------------------------

                # SMOTEENN is applied ONLY to training data.
                #
                # The test set must retain its original class distribution
                # so that model evaluation represents real-world performance.
                logging.info(
                    "Applying SMOTEENN to training data"
                )

                smt = SMOTEENN(
                    sampling_strategy="minority"
                )

                input_feature_train_final, target_feature_train_final = (
                    smt.fit_resample(
                        input_feature_train_arr,
                        target_feature_train_df
                    )
                )

                logging.info(
                    "Successfully applied SMOTEENN to training data"
                )

                # ------------------------------------------------------------------
                # 15. Keep the original testing data
                # ------------------------------------------------------------------

                # DO NOT apply SMOTEENN to test data.
                input_feature_test_final = input_feature_test_arr
                target_feature_test_final = target_feature_test_df

                logging.info(
                    "Testing data kept in its original class distribution"
                )

                # ------------------------------------------------------------------
                # 16. Combine transformed features with target
                # ------------------------------------------------------------------

                # Append the target column to the transformed training array.
                train_arr = np.c_[
                    input_feature_train_final,
                    np.array(target_feature_train_final)
                ]

                # Append the original test target to transformed test features.
                test_arr = np.c_[
                    input_feature_test_final,
                    np.array(target_feature_test_final)
                ]

                logging.info(
                    "Created final transformed train and test arrays"
                )

                # ------------------------------------------------------------------
                # 17. Save preprocessing object
                # ------------------------------------------------------------------

                # Save the fitted preprocessor so that the exact same
                # transformations can be applied during inference/deployment.
                save_object(
                    self.data_transformation_config
                    .transformed_object_file_path,
                    preprocessor
                )

                logging.info(
                    "Saved fitted preprocessing object"
                )

                # ------------------------------------------------------------------
                # 18. Save transformed training data
                # ------------------------------------------------------------------

                save_numpy_array_data(
                    self.data_transformation_config
                    .transformed_train_file_path,
                    array=train_arr
                )

                # ------------------------------------------------------------------
                # 19. Save transformed testing data
                # ------------------------------------------------------------------

                save_numpy_array_data(
                    self.data_transformation_config
                    .transformed_test_file_path,
                    array=test_arr
                )

                logging.info(
                    "Saved transformed train and test datasets"
                )

                # ------------------------------------------------------------------
                # 20. Create DataTransformationArtifact
                # ------------------------------------------------------------------

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=(
                        self.data_transformation_config
                        .transformed_object_file_path
                    ),
                    transformed_train_file_path=(
                        self.data_transformation_config
                        .transformed_train_file_path
                    ),
                    transformed_test_file_path=(
                        self.data_transformation_config
                        .transformed_test_file_path
                    )
                )

                logging.info(
                    "Data transformation completed successfully"
                )

                # Return artifact for the next pipeline component.
                return data_transformation_artifact

            else:

                # Stop the pipeline if data validation failed.
                raise Exception(
                    self.data_validation_artifact.message
                )

        except Exception as e:

            # Wrap the original error using the project's custom exception
            # so that the error contains useful system information.
            raise USvisaException(e, sys) from e