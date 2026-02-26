import cv2
import numpy as np


class Drawer:
    def __init__(self):
        self.canvas = None
        self.prev_x = None
        self.prev_y = None

    def initialize_canvas(self, frame):
        if self.canvas is None:
            self.canvas = np.zeros_like(frame)

    def draw(self, frame, x, y):
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y

        cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), (255, 0, 0), 5)
        self.prev_x, self.prev_y = x, y

    def reset_position(self):
        self.prev_x = None
        self.prev_y = None

    def get_output(self, frame):
        return cv2.add(frame, self.canvas)
