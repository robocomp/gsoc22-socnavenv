import cv2
import numpy as np
from socnavenv.envs.utils.object import Object
from socnavenv.envs.utils.human import Human
from socnavenv.envs.utils.utils import w2px, w2py
from math import atan2
from typing import List


class Human_Human_Interaction:
    """
    Class for Human-Human Interactions
    """

    def __init__(self, x, y, type:str, numOfHumans:int, radius:float, human_width) -> None:
        # center of interaction
        self.x = x
        self.y = y
        self.name = "human-human-interaction"
        assert numOfHumans >= 2, "Need at least 2 humans to interact"       
        if type != "moving" and type != "stationary":
            raise AssertionError("type should be \"moving\" or \"stationary\"")
        
        # indicates the type of interaction, whether it is moving or stationary
        self.type = type

        # radius of the interaction space
        self.radius = radius

        self.humans:List[Human] = []
        
        for _ in range(numOfHumans):
            if self.type == "stationary":
                self.add_human(Human(speed=0, width=human_width))
            else:
                self.add_human(Human(speed=0.1, width=human_width))
        
        # arranging all the humans around a circle
        self.arrange_humans()

        # counting the stopped time for moving interactions
        self.stopped_time = 0
    
    def add_human(self, h:Human):
        """
        Adding humans to the human list
        """
        self.humans.append(h)

    def arrange_humans(self):
        n = len(self.humans)
        theta = 0       
        increment = 2*(np.pi)/n

        if self.type == "moving":
            # theta chosen randomly between -pi to pi
            orientation = (np.random.random()-0.5) * np.pi * 2
    
        for i in range(n):
            h = self.humans[i]
            h.x = self.x + self.radius * np.cos(theta)
            h.y = self.x + self.radius * np.sin(theta)
            
            if self.type == "stationary":
                # humans would face the center as if talking to each other
                h.orientation = theta - np.pi

            elif self.type == "moving":
                # humans moving in the same direction, in a direction one direction
                h.orientation = orientation
            
            theta += increment
            if theta >= np.pi: theta -= 2*np.pi

    def collides(self, obj:Object):
        """
        To check for collision of any interacting human with another object
        """
        if obj.name == "human-human-interaction":
            for h1 in self.humans:
                for h2 in obj.humans:
                    if(h1.collides(h2)):
                        return True
            return False
        
        elif obj.name == "human-laptop-interaction":
            for h1 in self.humans:
                if(h1.collides(obj.human)):
                    return True
            return False

        for h in self.humans:
            if h.collides(obj): return True
        return False

    def update(self, time):
        if self.type == "stationary":
            pass

        elif self.type == "moving":
            for h in self.humans:
                h.update(time)
            
            for h in self.humans:
                if h.speed == 0:
                    self.stopped_time += 1
                    break
                else: self.stopped_time = 0

            # changing orientation
            if self.stopped_time == 10:
                for h in self.humans:
                    h.speed = -0.1
                    self.stopped_time = 0
                    self.update(time)
                orientation = (np.random.random()-0.5) * np.pi * 2
                for h in self.humans:
                    h.orientation = orientation
                    h.speed = 0.1
                self.stopped_time = 0

    def draw(self, img, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_SIZE_X, MAP_SIZE_Y):
        
        for h in self.humans:
            h.draw(img, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_SIZE_X, MAP_SIZE_Y)
        
        points = []
        for h in self.humans:
            points.append(
                [
                    w2px(h.x, PIXEL_TO_WORLD_X, MAP_SIZE_X),
                    w2py(h.y, PIXEL_TO_WORLD_Y, MAP_SIZE_Y)
                ]
            )
        points = np.array(points).reshape((-1,1,2))
        cv2.polylines(img, [np.int32(points)], True, (0, 0, 255), 1)
    