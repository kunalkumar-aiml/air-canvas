import cv2
import numpy as np


class Drawer:

    def __init__(self):

        self.canvas = None
        self.prev_x = None
        self.prev_y = None

        self.color = (255,0,0)
        self.brush_size = 5

        self.shape_mode = None

    def initialize_canvas(self, frame):

        if self.canvas is None:
            self.canvas = np.zeros_like(frame)

    def draw(self,x,y):

        if self.prev_x is None:
            self.prev_x,self.prev_y = x,y

        cv2.line(
            self.canvas,
            (self.prev_x,self.prev_y),
            (x,y),
            self.color,
            self.brush_size
        )

        self.prev_x,self.prev_y = x,y

    def erase(self,x,y):

        if self.prev_x is None:
            self.prev_x,self.prev_y = x,y

        cv2.line(
            self.canvas,
            (self.prev_x,self.prev_y),
            (x,y),
            (0,0,0),
            30
        )

        self.prev_x,self.prev_y = x,y

    def reset(self):

        self.prev_x = None
        self.prev_y = None

    def set_color(self,color):

        self.color = color

    def set_brush(self,size):

        self.brush_size = size

    def draw_shape(self,x1,y1,x2,y2):

        if self.shape_mode == "rectangle":

            cv2.rectangle(self.canvas,(x1,y1),(x2,y2),self.color,3)

        elif self.shape_mode == "circle":

            radius = int(((x2-x1)**2 + (y2-y1)**2)**0.5)

            cv2.circle(self.canvas,(x1,y1),radius,self.color,3)

    def output(self,frame):

        return cv2.add(frame,self.canvas)
