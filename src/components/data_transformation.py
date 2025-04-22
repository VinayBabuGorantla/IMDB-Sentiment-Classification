import os
import sys
import numpy as np
from dataclasses import dataclass
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer  # Optional if needed elsewhere

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    maxlen: int = 200  # max sequence length
    tokenizer_obj_path: str = os.path.join("artifacts", "tokenizer.pkl")  # Optional, for custom tokenization
    transformed_train_path: str = os.path.join("artifacts", "transformed_train.npz")
    transformed_test_path: str = os.path.join("artifacts", "transformed_test.npz")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path: str, test_path: str):
        logging.info("Started data transformation component for IMDB dataset.")
        try:
            # Load data
            train_data = np.load(train_path, allow_pickle=True)
            test_data = np.load(test_path, allow_pickle=True)

            X_train, y_train = train_data['X_train'], train_data['y_train']
            X_test, y_test = test_data['X_test'], test_data['y_test']

            logging.info(f"Loaded train and test data from {train_path} and {test_path}")
            logging.info(f"Original sequence lengths - Train[0]: {len(X_train[0])}, Test[0]: {len(X_test[0])}")

            # Pad sequences
            X_train_padded = pad_sequences(X_train, maxlen=self.config.maxlen, padding='post', truncating='post')
            X_test_padded = pad_sequences(X_test, maxlen=self.config.maxlen, padding='post', truncating='post')

            logging.info(f"Padded sequences to maxlen={self.config.maxlen}")

            # Save transformed data
            np.savez_compressed(self.config.transformed_train_path, X=X_train_padded, y=y_train)
            np.savez_compressed(self.config.transformed_test_path, X=X_test_padded, y=y_test)

            logging.info(f"Transformed train data saved at {self.config.transformed_train_path}")
            logging.info(f"Transformed test data saved at {self.config.transformed_test_path}")

            return self.config.transformed_train_path, self.config.transformed_test_path

        except Exception as e:
            raise CustomException(e, sys)
