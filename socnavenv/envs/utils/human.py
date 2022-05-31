import cv2
import numpy as np
from socnavenv.envs.utils.object import Object
from socnavenv.envs.utils.utils import w2px, w2py


class Human(Object):
    """
    Class for humans
    """

    def __init__(self, x=None, y=None, theta=None, width=None, speed=None) -> None:
        super().__init__("human")
        self.width = None
        self.is_static = False  # humans can move, so is_static is False
        self.speed = 0  # linear speed
        self.set(x, y, theta, width, speed)

    def set(self, x, y, theta, width, speed):
        super().set(x, y, theta)
        self.width = width
        if self.width is not None:
            self.length = width * 0.2  # thickness of the shoulder (for visualization)
            self.radius = width / 5  # radius of head (for visualization)
        if speed is not None:
            self.speed = speed  # speed

    def update(self, time):
        """
        For updating the coordinates of the human
        """
        assert (
            self.x != None and self.y != None and self.orientation != None
        ), "Coordinates or orientation are None type"
        moved = time * self.speed  # distance moved = speed x time
        self.x += moved * np.cos(self.orientation)  # updating x position
        self.y += moved * np.sin(self.orientation)  # updating y position

    def draw(self, img, PIXEL_TO_WORLD, MAP_SIZE):
        if self.color == None:
            color = (240, 114, 66)  # blue
        else:
            color = self.color
        assert self.width != None, "Width is None type."
        assert (
            self.x != None and self.y != None and self.orientation != None
        ), "Coordinates or orientation are None type"

        # p1, p2, p3, p4 are the coordinates of the corners of the rectangle. calculation is done so as to orient the rectangle at an angle.

        p1 = [
            w2px(
                (
                    self.x
                    + self.length / 2 * np.cos(self.orientation)
                    - self.width / 2 * np.sin(self.orientation)
                ),
                PIXEL_TO_WORLD,
                MAP_SIZE,
            ),
            w2py(
                (
                    self.y
                    + self.length / 2 * np.sin(self.orientation)
                    + self.width / 2 * np.cos(self.orientation)
                ),
                PIXEL_TO_WORLD,
                MAP_SIZE,
            ),
        ]

        p2 = [
            w2px(
                (
                    self.x
                    + self.length / 2 * np.cos(self.orientation)
                    + self.width / 2 * np.sin(self.orientation)
                ),
                PIXEL_TO_WORLD,
                MAP_SIZE,
            ),
            w2py(
                (
                    self.y
                    + self.length / 2 * np.sin(self.orientation)
                    - self.width / 2 * np.cos(self.orientation)
                ),
                PIXEL_TO_WORLD,
                MAP_SIZE,
            ),
        ]

        p3 = [
            w2px(
                (
                    self.x
                    - self.length / 2 * np.cos(self.orientation)
                    + self.width / 2 * np.sin(self.orientation)
                ),
                PIXEL_TO_WORLD,
                MAP_SIZE,
            ),
            w2py(
                (
                    self.y
                    - self.length / 2 * np.sin(self.orientation)
                    - self.width / 2 * np.cos(self.orientation)
                ),
                PIXEL_TO_WORLD,
                MAP_SIZE,
            ),
        ]

        p4 = [
            w2px(
                (
                    self.x
                    - self.length / 2 * np.cos(self.orientation)
                    - self.width / 2 * np.sin(self.orientation)
                ),
                PIXEL_TO_WORLD,
                MAP_SIZE,
            ),
            w2py(
                (
                    self.y
                    - self.length / 2 * np.sin(self.orientation)
                    + self.width / 2 * np.cos(self.orientation)
                ),
                PIXEL_TO_WORLD,
                MAP_SIZE,
            ),
        ]
        points = np.array([p1, p2, p3, p4])
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(
            img, [np.int32(points)], color
        )  # filling the rectangle made from the points with the specified color
        cv2.polylines(
            img, [np.int32(points)], True, (0, 0, 0), 2
        )  # bordering the rectangle

        black = (0, 0, 0)  # color for the head
        assert self.radius != None, "Radius is None type."
        assert self.x != None and self.y != None, "Coordinates are None type"

        radius = w2px(self.x + self.radius, PIXEL_TO_WORLD, MAP_SIZE) - w2px(
            self.x, PIXEL_TO_WORLD, MAP_SIZE
        )  # calculating no. of pixels corresponding to the radius

        cv2.circle(
            img,
            (
                w2px(
                    self.x + (self.width / 10) * np.cos(self.orientation),
                    PIXEL_TO_WORLD,
                    MAP_SIZE,
                ),
                w2py(
                    self.y + (self.width / 10) * np.sin(self.orientation),
                    PIXEL_TO_WORLD,
                    MAP_SIZE,
                ),
            ),
            radius,
            black,
            -1,
        )  # drawing a circle for the head of the human
