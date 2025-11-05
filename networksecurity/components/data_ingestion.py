import os
import sys
import numpy as np
import pandas as pd
from typing import Collection, List
from pymongo.database import Database
from sklearn.model_selection import train_test_split
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv('MONGO_DB_URL')

import certifi
ca = certifi.where()
from pymongo import MongoClient
from pymongo.server_api import ServerApi

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_into_dataframe(self) -> pd.DataFrame:
        """
        Export data from MongoDB to Pandas DataFrame
        """
        try:
            database_name: str = self.data_ingestion_config.database_name
            collection_name: str = self.data_ingestion_config.collection_name
            self.client: MongoClient = MongoClient(MONGO_DB_URL, server_api=ServerApi('1'), tlsCAFile=ca)
            db: Database = self.client[database_name]
            collection: Collection = db[collection_name]
            dataframe: pd.DataFrame = pd.DataFrame(list(collection.find()))
            if '_id' in dataframe.columns:
                dataframe.drop(columns=['_id'], inplace=True)
            dataframe.replace({'na': np.nan}, inplace=True)
            logger.info(f"Exported data from MongoDB to Pandas DataFrame: {dataframe.head()}")
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def save_data_into_feature_store(self, dataframe: pd.DataFrame) -> None:
        try:
            feature_store_file_path: str = self.data_ingestion_config.feature_store_file_path
            dir_path: str = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            train_file_dir: str = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(train_file_dir, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logger.info(f"Saved data into feature store: {feature_store_file_path}")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            train_set.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)
            logger.info(f"Split data into train and test sets: {train_set.shape}, {test_set.shape}")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe: pd.DataFrame = self.export_data_into_dataframe()
            self.save_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            data_ingestion_artifact: DataIngestionArtifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
