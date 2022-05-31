import cv2
import numpy as np
from socnavenv.envs.utils.object import Object
from socnavenv.envs.utils.utils import w2px, w2py

class Robot(Object):
    def __init__(self, x=None, y=None, theta=None, radius=None) -> None:
        super().__init__("robot")
        self.is_static = False
        self.radius = None  # radius of the robot
        self.set(x, y, theta, radius)

    def set(self, x, y, theta, radius):
        super().set(x, y, theta)
        self.radius = radius  

    def update(self, adv, rot, time):
        """
        For updating the coordinates of the robot.
        Input: adv : [-MAX_ADVANCE, +MAX_ADVANCE]
                rot: [-MAX_ROTATION, +MAX_ROTATION]
                time : float representing the time passed
        """
        self.x += adv*time*np.cos(self.orientation)  # updating the x-coordinate
        self.y += adv*time*np.sin(self.orientation)  # updating the y-coordinate
        self.orientation += rot*time  # updating the robot orientation

    def draw(self, img, PIXEL_TO_WORLD, MAP_SIZE):
        black = (0,0,0) 
        assert self.radius != None, "Radius is None type."
        assert self.x != None and self.y != None, "Coordinates are None type"

        radius = w2px(self.x + self.radius, PIXEL_TO_WORLD, MAP_SIZE) - w2px(
            self.x, PIXEL_TO_WORLD, MAP_SIZE
        )  # calculating no. of pixels corresponding to the radius
       
        cv2.circle(
            img,
            (
                w2px(self.x, PIXEL_TO_WORLD, MAP_SIZE),
                w2py(self.y, PIXEL_TO_WORLD, MAP_SIZE),
            ),
            radius,
            black,
            -1,
        )  # drawing a black circle for the robot
        
        left = (
            w2px(self.x + self.radius*0.35*np.cos(self.orientation + np.pi/2), PIXEL_TO_WORLD, MAP_SIZE),
            w2py(self.y + self.radius*0.35*np.sin(self.orientation + np.pi/2), PIXEL_TO_WORLD, MAP_SIZE)
        )

        right = (
            w2px(self.x + self.radius*0.35*np.cos(self.orientation - np.pi/2), PIXEL_TO_WORLD, MAP_SIZE),
            w2py(self.y + self.radius*0.35*np.sin(self.orientation - np.pi/2), PIXEL_TO_WORLD, MAP_SIZE)
        )

        front = (
            w2px(self.x + self.radius*0.35*np.cos(self.orientation), PIXEL_TO_WORLD, MAP_SIZE),
            w2py(self.y + self.radius*0.35*np.sin(self.orientation), PIXEL_TO_WORLD, MAP_SIZE)
        )

        center = (
            w2px(self.x, PIXEL_TO_WORLD, MAP_SIZE),
            w2py(self.y, PIXEL_TO_WORLD, MAP_SIZE)
        )

        # drawing lines to get sense of the orientation of the robot.
        cv2.line(img, left, right, (27, 194, 169), 8)
        cv2.line(img, center, front, (27, 194, 169), 8)
