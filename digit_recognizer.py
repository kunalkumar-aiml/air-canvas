import cv2
import numpy as np
import tensorflow as tf


class DigitRecognizer:

    def __init__(self):
        self.model = tf.keras.models.load_model("digit_model.h5")

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))
        normalized = resized / 255.0
        reshaped = normalized.reshape(1, 28, 28, 1)
        return reshaped

    def predict(self, image):
        processed = self.preprocess(image)
        prediction = self.model.predict(processed)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        return digit, confidence
