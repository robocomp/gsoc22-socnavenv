import cv2
import numpy as np
from socnavenv.envs.utils.object import Object
from socnavenv.envs.utils.utils import w2px, w2py
from math import atan2

class Robot(Object):
    def __init__(self, x=None, y=None, theta=None, radius=None, goal_x=None, goal_y=None) -> None:
        super().__init__("robot")
        self.is_static = False
        self.radius = None  # radius of the robot
        self.goal_x = None  # x-coordinate of the goal
        self.goal_y = None  # y-coordinate of the goal
        self.linear_vel = 0.0  # linear velocity
        self.angular_vel = 0.0  # angular velocity
        self.set(x, y, theta, radius, goal_x, goal_y)

    def set(self, x, y, theta, radius, goal_x, goal_y):
        super().set(x, y, theta)
        self.radius = radius  
        self.goal_x = goal_x
        self.goal_y = goal_y

    def update(self, time):
        """
        For updating the coordinates of the robot.
        Input: time : float representing the time passed
        """
        self.x += self.linear_vel*time*np.cos(self.orientation)  # updating the x-coordinate
        self.y += self.linear_vel*time*np.sin(self.orientation)  # updating the y-coordinate
        self.orientation += self.angular_vel*time  # updating the robot orientation

    def update_velocity(self, vel_x, vel_y, MAX_ADVANCE):
        # vel_orientation = vel_x * np.cos(self.orientation) + vel_y * np.sin(self.orientation)
        # self.linear_vel += vel_orientation
        # if self.linear_vel > MAX_ADVANCE:
        #     self.linear_vel = MAX_ADVANCE
        curr_vel_x = self.linear_vel * np.cos(self.orientation)
        curr_vel_y = self.linear_vel * np.sin(self.orientation)

        curr_vel_x += vel_x
        curr_vel_y += vel_y

        self.orientation = atan2(curr_vel_y, curr_vel_x)
        
        # self.linear_vel = np.sqrt((curr_vel_x)**2 + (curr_vel_y)**2)

        # if self.linear_vel > MAX_ADVANCE:
        #     self.linear_vel = MAX_ADVANCE

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
