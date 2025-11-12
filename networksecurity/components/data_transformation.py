import os
import sys
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constants.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from networksecurity.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact
)

from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        logger.info(f"Entered get_data_transformer_object method of DataTransformation class")
        try:
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            pipeline: Pipeline = Pipeline(steps=[
                ("imputer", imputer)
            ])
            return pipeline
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logger.info(f"Entered initiate_data_transformation method of DataTransformation class")
        try:
            logger.info(f"Loading train and test data as pandas dataframe")
            logger.info(f"Loading valid train file path: {self.data_validation_artifact.valid_train_file_path}")
            train_df = DataTransformation.read_data(file_path=self.data_validation_artifact.valid_train_file_path)
            logger.info(f"Loading valid test file path: {self.data_validation_artifact.valid_test_file_path}")
            test_df = DataTransformation.read_data(file_path=self.data_validation_artifact.valid_test_file_path)

            # training dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1,0)

            # testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1,0)

            preprocessor: Pipeline = self.get_data_transformer_object()
            input_feature_train_df = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_df = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_df, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_df, np.array(target_feature_test_df)]

            save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_file_path, array=test_arr)

            logger.info(f"Saved transformed train array to {self.data_transformation_config.transformed_train_file_path}")
            logger.info(f"Saved transformed test array to {self.data_transformation_config.transformed_test_file_path}")

            save_object(file_path=self.data_transformation_config.transformed_object_file_path, obj=preprocessor)
            logger.info(f"Saved transformed object to {self.data_transformation_config.transformed_object_file_path}")

            # preparing artifacts
            data_transformation_artifact: DataTransformationArtifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )
            return data_transformation_artifact


        except Exception as e:
            raise NetworkSecurityException(e, sys)