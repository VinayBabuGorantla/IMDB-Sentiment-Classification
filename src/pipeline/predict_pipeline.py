import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 3 = only errors

import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.logger import logging
from src.exception import CustomException

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "sentiment_model.h5")
        self.maxlen = 200
        self.vocab_size = 10000  # Add this line
        self.word_index = self._get_imdb_word_index()


    def _get_imdb_word_index(self):
        from tensorflow.keras.datasets import imdb
        return imdb.get_word_index()

    def preprocess_review(self, review: str):
        try:
            review = review.lower().split()
            sequence = []
            for word in review:
                idx = self.word_index.get(word, 2) + 3  # OOV index offset
                if idx >= self.vocab_size:  # Only keep indices the model can handle
                    idx = 2  # Replace with OOV token index
                sequence.append(idx)

            sequence = pad_sequences([sequence], maxlen=self.maxlen, padding='post', truncating='post')
            return sequence
        except Exception as e:
            raise CustomException(e, sys)


    def predict(self, review: str):
        try:
            logging.info("Prediction pipeline started.")
            model = load_model(self.model_path)
            logging.info("Model loaded successfully.")

            preprocessed = self.preprocess_review(review)
            prediction = model.predict(preprocessed)[0][0]
            sentiment = "Positive" if prediction >= 0.5 else "Negative"

            logging.info(f"Prediction complete. Sentiment: {sentiment}, Score: {prediction:.4f}")
            return sentiment, prediction
        except Exception as e:
            raise CustomException(e, sys)
