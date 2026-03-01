import cv2
import numpy as np
import time
import config


class Drawer:
    def __init__(self):
        self.canvas = None
        self.prev_x = None
        self.prev_y = None
        self.color = config.COLORS["blue"]
        self.last_save_time = 0
        self.erase_mode = False

    def initialize_canvas(self, frame):
        if self.canvas is None:
            self.canvas = np.zeros_like(frame)

    def smooth_point(self, x, y, alpha=0.5):
        if self.prev_x is None:
            return x, y
        smooth_x = int(self.prev_x * (1 - alpha) + x * alpha)
        smooth_y = int(self.prev_y * (1 - alpha) + y * alpha)
        return smooth_x, smooth_y

    def draw(self, x, y):
        x, y = self.smooth_point(x, y)

        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y

        if self.erase_mode:
            cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y),
                     (0, 0, 0), config.ERASER_THICKNESS)
        else:
            cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y),
                     self.color, config.BRUSH_THICKNESS)

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

    def set_color(self, color):
        self.erase_mode = False
        self.color = color

    def enable_eraser(self):
        self.erase_mode = True

    def get_output(self, frame):
        return cv2.add(frame, self.canvas)
