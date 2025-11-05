from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataIngestionConfig, DataValidationConfig, TrainingPipelineConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger

if __name__ == "__main__":
    import sys
    try:
        training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()
        data_ingestion_config: DataIngestionConfig = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion: DataIngestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact: DataIngestionArtifact = data_ingestion.initiate_data_ingestion()
        logger.info(f"Data ingestion artifact: {data_ingestion_artifact}")

        data_validation_config: DataValidationConfig = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
        data_validation_artifact: DataValidationArtifact = data_validation.initiate_data_validation()
        logger.info(f"Data validation artifact: {data_validation_artifact}")
    except Exception as e:
        raise NetworkSecurityException(e, sys)
