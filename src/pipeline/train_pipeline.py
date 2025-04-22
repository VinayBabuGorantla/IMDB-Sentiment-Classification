import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 3 = only errors

import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException

if __name__ == "__main__":
    try:
        logging.info(">>> Training pipeline started.")

        # Step 1: Ingest data
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        # Step 2: Transform data
        transformer = DataTransformation()
        transformed_train_path, transformed_test_path = transformer.initiate_data_transformation(train_path, test_path)

        # Step 3: Train model
        trainer = ModelTrainer()
        final_accuracy = trainer.initiate_model_trainer(transformed_train_path, transformed_test_path)

        logging.info(f"Training pipeline completed successfully with final val accuracy: {final_accuracy:.4f}")

    except Exception as e:
        raise CustomException(e, sys)
