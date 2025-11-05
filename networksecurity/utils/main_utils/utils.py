import yaml
import sys
import os
# import pickle
# import dill
# import numpy as np

import networksecurity.exception.exception as NetworkSecurityException
from networksecurity.logging.logger import logger

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        with open(file_path, 'w') as yaml_file:
            yaml.dump(content, yaml_file)
        logger.info(f"File {file_path} created successfully")
    except Exception as e:
        raise NetworkSecurityException(e, sys)