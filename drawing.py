import cv2
import numpy as np
import time


class Drawer:
    def __init__(self):
        self.canvas = None
        self.prev_x = None
        self.prev_y = None
        self.color = (255, 0, 0)
        self.last_save_time = 0

    def initialize_canvas(self, frame):
        if self.canvas is None:
            self.canvas = np.zeros_like(frame)

    def draw(self, x, y):
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y

        cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), self.color, 6)
        self.prev_x, self.prev_y = x, y

    def reset_position(self):
        self.prev_x = None
        self.prev_y = None

    def clear_canvas(self):
        if self.canvas is not None:
            self.canvas[:] = 0

    def save_canvas(self):
        current_time = time.time()
        if current_time - self.last_save_time > 1:
            filename = f"drawing_{int(current_time)}.png"
            cv2.imwrite(filename, self.canvas)
            self.last_save_time = current_time
            print(f"Saved: {filename}")

    def set_color(self, color):
        self.color = color

    def get_output(self, frame):
        return cv2.add(frame, self.canvas)
