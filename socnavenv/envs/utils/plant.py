import cv2
import numpy as np
from socnavenv.envs.utils.object import Object
from socnavenv.envs.utils.utils import w2px, w2py


class Plant(Object):
    """
    Class for Plant
    """

    def __init__(self, x=None, y=None, radius=None) -> None:
        super().__init__("plant")
        self.radius = None # radius of the plant
        self.set(x, y, 0, radius)

    def set(self, x, y, theta, radius):
        super().set(x, y, theta)
        self.radius = radius

    def draw(self, img, PIXEL_TO_WORLD, MAP_SIZE):
        brown = (29, 67, 105)  # brown
        green = (0, 200, 0) # green
        assert self.radius != None, "Radius is None type."
        assert self.x != None and self.y != None, "Coordinates are None type"

        radius = w2px(self.x + self.radius, PIXEL_TO_WORLD, MAP_SIZE) - w2px(
            self.x, PIXEL_TO_WORLD, MAP_SIZE
        ) # calculating the number of pixels corresponding to the radius

        cv2.circle(
            img,
            (
                w2px(self.x, PIXEL_TO_WORLD, MAP_SIZE),
                w2py(self.y, PIXEL_TO_WORLD, MAP_SIZE),
            ),
            radius,
            brown,
            -1,
        ) # drawing a brown circle for the pot in which the plant is kept
        cv2.circle(
            img,
            (
                w2px(self.x, PIXEL_TO_WORLD, MAP_SIZE),
                w2py(self.y, PIXEL_TO_WORLD, MAP_SIZE),
            ),
            int(radius / 2),
            green,
            -1,
        )  # drawing a green circle for the plant
