from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException
import sys
from sklearn.metrics import f1_score, precision_score, recall_score


def get_classification_score(y_true: list, y_pred: list) -> ClassificationMetricArtifact:
    try:
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        return ClassificationMetricArtifact(
            f1_score=f1,
            precision_score=precision,
            recall_score=recall
        )
    except Exception as e:
        raise NetworkSecurityException(e, sys)