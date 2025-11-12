import yaml
import sys
import os
import numpy as np
# import dill
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

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

def save_numpy_array_data(file_path: str, array: np.array) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            np.save(file, array)
        logger.info(f"File {file_path} created successfully")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def save_object(file_path: str, obj: object) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logger.info(f"File {file_path} created successfully")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def load_object(file_path: str) -> object:
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, 'rb') as file:
            return np.load(file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def evaluate_models(X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array, models: dict, params: dict) -> dict:
    try:
        report: dict = {}
        for model_name, model in models.items():
            # model.set_params(**params[model_name])
            para = params[model_name]

            gs = GridSearchCV(model, para, cv=3, verbose=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # report[model_name] = {
            #     'train_score': train_model_score,
            #     'test_score': test_model_score,
            #     'best_model': gs.best_params_
            # }

            report[model_name] = test_model_score
            logger.info(f"Model Name: {model_name}, Model Score: {test_model_score}")
        return report
    except Exception as e:
        raise NetworkSecurityException(e, sys)