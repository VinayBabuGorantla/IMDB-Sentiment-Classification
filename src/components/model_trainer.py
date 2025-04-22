import os
import sys
from dataclasses import dataclass
import numpy as np
from typing import Literal

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import mlflow
import mlflow.tensorflow

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "sentiment_model.h5")
    vocab_size: int = 10000
    embedding_dim: int = 128
    maxlen: int = 200
    rnn_type: Literal['lstm', 'gru'] = 'lstm'  # switch between 'lstm' or 'gru'

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def build_model(self):
        logging.info(f"Building model using {self.config.rnn_type.upper()} architecture.")
        model = Sequential()
        model.add(Embedding(self.config.vocab_size, self.config.embedding_dim, input_length=self.config.maxlen))

        if self.config.rnn_type == 'lstm':
            model.add(LSTM(128))
        else:
            model.add(GRU(128))

        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        return model

    def initiate_model_trainer(self, train_array_path: str, test_array_path: str):
        logging.info("Started model training component.")
        try:
            # Load transformed data
            train_data = np.load(train_array_path, allow_pickle=True)
            test_data = np.load(test_array_path, allow_pickle=True)

            X_train, y_train = train_data['X'], train_data['y']
            X_test, y_test = test_data['X'], test_data['y']

            model = self.build_model()

            early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            # Enable MLflow autologging
            mlflow.tensorflow.autolog()

            with mlflow.start_run():
                history = model.fit(
                    X_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop]
                )

                # Save model
                model.save(self.config.model_path)
                logging.info(f"Model trained and saved at {self.config.model_path}")

                # Return final val accuracy
                final_accuracy = history.history['val_accuracy'][-1]
                logging.info(f"Final validation accuracy: {final_accuracy:.4f}")
                return final_accuracy

        except Exception as e:
            raise CustomException(e, sys)
