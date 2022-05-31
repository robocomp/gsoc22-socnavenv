import cv2
import numpy as np
from socnavenv.utils.object import Object
from socnavenv.utils.utils import w2px, w2py

class Wall(Object):
    """
    Class for Wall
    """

    def __init__(self, x=None, y=None, theta=None, length=None) -> None:
        super().__init__("wall")
        self.length = None  # length of the wall
        self.set(x, y, theta, length)

    def set(self, x, y, theta, length):
        super().set(x, y, theta)
        self.length = length

    def draw(self, img, PIXEL_TO_WORLD, MAP_SIZE):
        if self.color == None:
            color = (0, 0, 0)  # black
        else: color = self.color
        
        assert self.length != None, "Length is None type."
        assert (
            self.x != None and self.y != None and self.orientation != None
        ), "Coordinates or orientation are None type"


        # p1, p2 are the end points of the wall. (self.x and self.y are the center of the wall)
        p1 = (
            w2px(
                (self.x + self.length / 2 * np.cos(self.orientation)),
                PIXEL_TO_WORLD,
                MAP_SIZE,
            ),
            w2py(
                (self.y + self.length / 2 * np.sin(self.orientation)),
                PIXEL_TO_WORLD,
                MAP_SIZE,
            ),
        )

        p2 = (
            w2px(
                (self.x - self.length / 2 * np.cos(self.orientation)),
                PIXEL_TO_WORLD,
                MAP_SIZE,
            ),
            w2py(
                (self.y - self.length / 2 * np.sin(self.orientation)),
                PIXEL_TO_WORLD,
                MAP_SIZE,
            ),
        )

        cv2.line(img, p2, p1, color, 5) # drawing line for the wall
