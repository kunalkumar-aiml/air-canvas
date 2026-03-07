import cv2
import numpy as np
import time


class Drawer:

    def __init__(self):

        self.canvas=None
        self.prev_x=None
        self.prev_y=None

        self.color=(255,0,0)
        self.brush_size=5

        self.start_shape=None
        self.shape_mode=None

        self.last_save=0


    def initialize(self,frame):

        if self.canvas is None:
            self.canvas=np.zeros_like(frame)


    def draw(self,x,y):

        if self.prev_x is None:
            self.prev_x,self.prev_y=x,y

        cv2.line(
            self.canvas,
            (self.prev_x,self.prev_y),
            (x,y),
            self.color,
            self.brush_size
        )

        self.prev_x,self.prev_y=x,y


    def erase(self,x,y):

        if self.prev_x is None:
            self.prev_x,self.prev_y=x,y

        cv2.line(
            self.canvas,
            (self.prev_x,self.prev_y),
            (x,y),
            (0,0,0),
            40
        )

        self.prev_x,self.prev_y=x,y


    def reset(self):

        self.prev_x=None
        self.prev_y=None


    def set_color(self,color):

        self.color=color


    def set_brush(self,size):

        self.brush_size=size


    def clear(self):

        if self.canvas is not None:
            self.canvas[:]=0


    def save(self):

        current=time.time()

        if current-self.last_save>1:

            name=f"drawing_{int(current)}.png"

            cv2.imwrite(name,self.canvas)

            self.last_save=current


    def output(self,frame):

        return cv2.add(frame,self.canvas)
