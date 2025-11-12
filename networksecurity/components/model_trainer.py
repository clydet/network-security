import os
import sys
import numpy as np
# import pandas as pd

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger

from networksecurity.entity.artifact_entity import (
    ClassificationMetricArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.main_utils.utils import (
    save_object, load_object, load_numpy_array_data, evaluate_models
)
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
import mlflow

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, model: object, metric: ClassificationMetricArtifact):
        with mlflow.start_run():
            f1_score = metric.f1_score
            precision_score = metric.precision_score
            recall_score = metric.recall_score
            mlflow.log_metric("F1 Score", f1_score)
            mlflow.log_metric("Precision Score", precision_score)
            mlflow.log_metric("Recall Score", recall_score)
            mlflow.sklearn.log_model(model, "model")

    def train_model(self, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array) -> object:
        try:
            models = {
                "Logistic Regression": LogisticRegression(verbose=1),
                # "KNeighbors Classifier": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(verbose=1),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "AdaBoost": AdaBoostClassifier()
            }
            params = {
                "Logistic Regression": {
                    # "C": [0.001, 0.01, 0.1, 1, 10, 100],
                    # "penalty": ["l1", "l2"],
                    # "solver": ["liblinear", "saga"],
                },
                "Decision Tree": {
                    'criterion': ['gini', 'entropy','log_loss'],
                    # "max_depth": [3, 5, 7, 9],
                    # "min_samples_split": [2, 3, 4, 5],
                    # "min_samples_leaf": [1, 2, 3, 4],
                },
                "Random Forest": {
                    # "n_estimators": [8,16,32,64,128,256],
                    # "max_depth": [3, 5, 7, 9],
                },
                "Gradient Boosting": {
                    # "learning_rate": [0.1, 0.05, 0.01, .001],
                    "learning_rate": [0.1, 0.05, .001],
                    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9],
                    "n_estimators": [8,16,32,64,128,256],
                    # "n_estimators": [100, 200, 300, 400, 500],
                    # "max_depth": [3, 5, 7, 9],
                },
                "AdaBoost": {
                    "n_estimators": [8,16,32,64,128,256],
                    "learning_rate": [0.001, 0.01, 0.1, 1, 10, 100],
                }
            }

            model_report: dict = evaluate_models(
                X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test,
                models=models, params=params)
            
            # to get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            logger.info(f"Best Model Name: {best_model_name}, Best Model Score: {best_model_score}")
            best_model = models[best_model_name]
            y_train_pred = best_model.predict(x_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            
            # track experiments with mlflow
            self.track_mlflow(best_model, classification_train_metric)

            y_test_pred = best_model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            # track experiments with mlflow
            self.track_mlflow(best_model, classification_test_metric)

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=network_model)

            ## Model Trainer Artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            logger.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(file_path=train_file_path)
            test_arr = load_numpy_array_data(file_path=test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            model = self.train_model(x_train, y_train, x_test, y_test)
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=model)
            return model

        except Exception as e:
            raise NetworkSecurityException(e, sys)

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import f1_score, precision_score, recall_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline

