from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file, save_object
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH

from scipy.stats import ks_2samp
import pandas as pd
import os
import sys
import yaml

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns: int = len(self._schema_config['columns'])
            logger.info(f"Number of columns: {number_of_columns}")
            logger.info(f"Number of columns in dataframe: {len(dataframe.columns)}")

            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_numerical_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            dataframe_columns = list(dataframe.columns)
            numerical_columns = self._schema_config['numerical_columns']
            numerical_columns_present = True
            missing_numerical_columns = []
            for column in numerical_columns:
                if column not in dataframe_columns:
                    numerical_columns_present = False
                    missing_numerical_columns.append(column)
            if not numerical_columns_present:
                error_message = f"Validation failed for numerical columns in dataframe. Missing numerical columns: {missing_numerical_columns}"
                raise NetworkSecurityException(error_message, sys)
            return True
        except NetworkSecurityException:
            raise
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                report[column] = {}
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if is_same_dist.pvalue > threshold:
                    status = False
                    report[column]['pvalue'] = float(is_same_dist.pvalue)
                    report[column]['drift_status'] = False
                else:
                    report[column]['pvalue'] = float(is_same_dist.pvalue)
                    report[column]['drift_status'] = True
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report, replace=True)
            return status
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_dataframe: pd.DataFrame = DataValidation.read_data(file_path=train_file_path)
            test_dataframe: pd.DataFrame = DataValidation.read_data(file_path=test_file_path)

            train_validation_status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not train_validation_status:
                error_message = f"Validation failed for number of columns in train dataframe. Expected: {len(self._schema_config['columns'])}, Found: {len(train_dataframe.columns)}"
                raise NetworkSecurityException(error_message, sys)

            test_validation_status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not test_validation_status:
                error_message = f"Validation failed for number of columns in test dataframe. Expected: {len(self._schema_config['columns'])}, Found: {len(test_dataframe.columns)}"
                raise NetworkSecurityException(error_message, sys)
            
            train_validation_status = self.validate_numerical_columns(dataframe=train_dataframe)
            if not train_validation_status:
                error_message = f"Validation failed for numerical columns in train dataframe. Expected: {len(self._schema_config['numerical_columns'])}, Found: {len(train_dataframe.columns)}"
                raise NetworkSecurityException(error_message, sys)

            test_validation_status = self.validate_numerical_columns(dataframe=test_dataframe)
            if not test_validation_status:
                error_message = f"Validation failed for numerical columns in test dataframe. Expected: {len(self._schema_config['numerical_columns'])}, Found: {len(test_dataframe.columns)}"
                raise NetworkSecurityException(error_message, sys)
            
            train_drift_status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)
            save_object(file_path=self.data_validation_config.drift_report_file_path, obj=train_drift_status)

            data_validation_artifact = DataValidationArtifact(
                validation_status=train_drift_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            return data_validation_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)