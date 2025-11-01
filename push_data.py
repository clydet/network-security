import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv('MONGO_DB_URL')
print(MONGO_DB_URL)

import certifi
ca = certifi.where()

import pandas as pd
import numpy as np
import pymongo

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger

class NetworkDataExtractor():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def cvs_to_json_converter(self, file_path: str) -> list[dict]:
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def push_data_to_mongo(self, records: list[dict], collection_name: str, db_name: str):
        try:
            self.collection_name = collection_name
            self.db_name = db_name
            self.records = records

            self.client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise NetworkSecurityException(e, sys)

if __name__ == "__main__":
    FILE_PATH = 'Network_Data/phishingData.csv'
    COLLECTION_NAME = 'NetworkData'
    DB_NAME = 'CLYDE'

    extractor = NetworkDataExtractor()
    records = extractor.cvs_to_json_converter(FILE_PATH)
    num_records = extractor.push_data_to_mongo(records, COLLECTION_NAME, DB_NAME)
    print(f'Number of records pushed to MongoDB: {num_records}')
