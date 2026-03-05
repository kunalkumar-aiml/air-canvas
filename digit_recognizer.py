import cv2
import numpy as np
import tensorflow as tf


class DigitRecognizer:

    def __init__(self):
        self.model = tf.keras.models.load_model("digit_model.h5")

    def preprocess(self, canvas):

        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        resized = cv2.resize(thresh, (28, 28))

        normalized = resized / 255.0

        reshaped = normalized.reshape(1, 28, 28, 1)

        return reshaped

    def predict(self, canvas):

        processed = self.preprocess(canvas)

        prediction = self.model.predict(processed, verbose=0)

        digit = int(np.argmax(prediction))

        confidence = float(np.max(prediction))

        return digit, confidence
