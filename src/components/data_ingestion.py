import os
import sys
from dataclasses import dataclass

from tensorflow.keras.datasets import imdb
import numpy as np

from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw_imdb_data.npz")
    train_data_path: str = os.path.join("artifacts", "train.npz")
    test_data_path: str = os.path.join("artifacts", "test.npz")
    num_words: int = 10000  # only top 10,000 most frequent words

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component for IMDB dataset.")
        try:
            # Load dataset from Keras
            (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=self.config.num_words)
            logging.info("IMDB dataset loaded successfully from Keras.")

            # Save raw dataset if needed later (X_train + y_train + X_test + y_test)
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            np.savez_compressed(self.config.raw_data_path, 
                                X_train=X_train, y_train=y_train, 
                                X_test=X_test, y_test=y_test)
            logging.info(f"Raw IMDB data saved at {self.config.raw_data_path}")

            # Save train and test separately for modular pipeline
            np.savez_compressed(self.config.train_data_path, X_train=X_train, y_train=y_train)
            np.savez_compressed(self.config.test_data_path, X_test=X_test, y_test=y_test)
            logging.info(f"Train and test data saved at {self.config.train_data_path} and {self.config.test_data_path}")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
