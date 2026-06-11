import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:

            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()

            logging.info("Starting data transformation")
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
                train_path, test_path)
            
            logging.info("Starting model training")
            model_trainer = ModelTrainer()
            score = model_trainer.initiate_model_trainer(
                train_arr, test_arr)

            logging.info(f"Training complete. Best accuracy: {score:.4f}")
            return score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    score = pipeline.run_pipeline()
    print(f"\nTraining Complete. Best Model Accuracy: {score:.4f}")
