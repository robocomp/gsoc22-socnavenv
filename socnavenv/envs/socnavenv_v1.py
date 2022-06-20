import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys
from gym import spaces
import cv2

from socnavenv.envs.utils.human import Human
from socnavenv.envs.utils.laptop import Laptop
from socnavenv.envs.utils.plant import Plant
from socnavenv.envs.utils.robot import Robot
from socnavenv.envs.utils.table import Table
from socnavenv.envs.utils.wall import Wall
from socnavenv.envs.utils.object import Object
from socnavenv.envs.utils.utils import w2px, w2py, uniform_circular_sampler


# rendering params
RESOLUTION = 700.
RESOLUTION_VIEW = 1000.
MILLISECONDS = 30

# episode params
MAX_TICKS = 200
TIMESTEP = 1


# rewards
REACH_REWARD = 1.0
OUTOFMAP_REWARD = -0.5
MAXTICKS_REWARD = -0.5
ALIVE_REWARD = 0.0
COLLISION_REWARD = -1.0
DISTANCE_REWARD_DIVISOR = 1000

# robot params
ROBOT_RADIUS = 0.7
GOAL_RADIUS = 0.5
GOAL_THRESHOLD = ROBOT_RADIUS + GOAL_RADIUS

# human params
HUMAN_DIAMETER = 0.72

# laptop params
LAPTOP_WIDTH=0.4
LAPTOP_LENGTH=0.6
LAPTOP_RADIUS = np.sqrt((LAPTOP_LENGTH/2)**2 + (LAPTOP_WIDTH/2)**2)

# plant params
PLANT_RADIUS = 0.4

# table params
TABLE_LENGTH = 3.0
TABLE_WIDTH = 1.5
TABLE_RADIUS = np.sqrt((TABLE_LENGTH/2)**2 + (TABLE_WIDTH/2)**2)

# wall params
WALL_THICKNESS = 0.2

assert(REACH_REWARD>0)
assert(OUTOFMAP_REWARD<0)
assert(MAXTICKS_REWARD<0)
# assert(ALIVE_REWARD<0)
assert(COLLISION_REWARD<0)
assert(DISTANCE_REWARD_DIVISOR>1)
assert(MAX_TICKS>1)
assert(GOAL_RADIUS>0)

DEBUG = 0
if 'debug' in sys.argv or "debug=2" in sys.argv:
    DEBUG = 2
elif "debug=1" in sys.argv:
    DEBUG = 1


# TO DO List
#
# Not urgent:
# - Improve how the robot moves (actual differential platform)
#


class SocNavEnv_v1(gym.Env):
    """
    Class for the environment
    """
    metadata = {}
    def __init__(self) -> None:
        super().__init__()
        self.window_initialised = False
        # the number of steps taken in the current episode
        self.ticks = 0  
        # humans in the environment
        self.humans = [] 
        # laptops in the environment
        self.laptops = [] 
        # walls in the environment
        self.walls = []  
        # plants in the environment
        self.plants = []  
        # tables in the environment
        self.tables = []  
        # all entities in the environment
        self.entities = None 
        
        # robot
        self.robot:Robot = None

        # environment parameters
        self.MARGIN = 0.5
        self.MAX_ADVANCE_HUMAN = 0.14
        self.MAX_ADVANCE_ROBOT = 0.1
        self.MAX_ROTATION = np.pi
        self.NUMBER_OF_WALLS = 4 
        # wall segment size
        self.WALL_SEGMENT_SIZE = 1.0


        # defining the max limit of entities
        self.MAX_HUMANS = 8
        self.MAX_TABLES = 3
        self.MAX_PLANTS = 5
        self.MAX_LAPTOPS = 4
           
        # flag parameter that controls whether padded observations will be returned or not
        self.get_padded_observations = False

        # to check if the episode has finished
        self.robot_is_done = True
        # for rendering the world to an OpenCV image
        self.world_image = np.zeros((int(RESOLUTION),int(RESOLUTION),3))
        
        # parameters for integrating multiagent particle environment's forces

        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

        # shape of the environment
        self.shape = None

        # to initialize the environment
        self.reset()

    def set_padded_observations(self, val:bool):
        """
        To assign True/False to the parameter get_padded_observations. True will indicate that padding will be done. Else padding 
        Args: val (bool): True/False value that would enable/disable padding in the observations received henceforth 
        """
        self.get_padded_observations = val

    def randomize_params(self):
        """
        To randomly initialize the number of entities of each type. Specifically, this function would initialize the MAP_SIZE, NUMBER_OF_HUMANS, NUMBER_OF_PLANTS, NUMBER_OF_LAPTOPS and NUMBER_OF_TABLES
        """
        self.MAP_X = np.random.randint(16, 25)
        
        if self.shape == "square":
            self.MAP_Y = self.MAP_X
        else :
            self.MAP_Y = np.random.randint(16, 25)
        
        # L_X sampled between MAP_X/4 and MAP_X*3/4
        self.L_X = (random.random() * (self.MAP_X/2)) + self.MAP_X/4 
        self.L_Y = (random.random() * (self.MAP_Y/2)) + self.MAP_Y/4
        self.RESOLUTION_X = int(1500 * self.MAP_X/(self.MAP_X + self.MAP_Y))
        self.RESOLUTION_Y = int(1500 * self.MAP_Y/(self.MAP_X + self.MAP_Y))
        self.NUMBER_OF_HUMANS = random.randint(3, self.MAX_HUMANS)  # number of humans in the env
        self.NUMBER_OF_PLANTS = random.randint(2, self.MAX_PLANTS)  # number of plants in the env
        self.NUMBER_OF_TABLES = random.randint(1, self.MAX_TABLES)  # number of tables in the env
        self.NUMBER_OF_LAPTOPS = random.randint(1, self.MAX_LAPTOPS)  # number of laptops in the env. Laptops will be sampled on tables

    @property
    def TOTAL_OBJECTS(self):
        return self.NUMBER_OF_HUMANS + self.NUMBER_OF_PLANTS + self.NUMBER_OF_TABLES + self.NUMBER_OF_LAPTOPS + self.NUMBER_OF_WALLS

    @property
    def PIXEL_TO_WORLD_X(self):
        return self.RESOLUTION_X / self.MAP_X

    @property
    def PIXEL_TO_WORLD_Y(self):
        return self.RESOLUTION_Y / self.MAP_Y
    
    @property
    def MAX_OBSERVATION_LENGTH(self):
        return (self.MAX_HUMANS + self.MAX_LAPTOPS + self.MAX_PLANTS + self.MAX_TABLES) * 13 + 8
    
    @property
    def observation_space(self):
        """
        Observation space includes the goal coordinates in the robot's frame and the relative coordinates and speeds (linear & angular) of all the objects in the scenario
        
        Returns:
        gym.spaces.Dict : the observation space of the environment
        """

        d = {

            "goal": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2)], dtype=np.float32), 
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2)], dtype=np.float32),
                shape=((self.robot.one_hot_encoding.shape[0]+2, )),
                dtype=np.float32

            ),

            "humans": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0, -HUMAN_DIAMETER/2, -(self.MAX_ADVANCE_HUMAN + self.MAX_ADVANCE_ROBOT), -self.MAX_ROTATION] * (self.MAX_HUMANS if self.get_padded_observations else self.NUMBER_OF_HUMANS), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, HUMAN_DIAMETER/2, +(self.MAX_ADVANCE_HUMAN + self.MAX_ADVANCE_ROBOT), +self.MAX_ROTATION] * (self.MAX_HUMANS if self.get_padded_observations else self.NUMBER_OF_HUMANS), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 7)*(self.MAX_HUMANS if self.get_padded_observations else self.NUMBER_OF_HUMANS),)),
                dtype=np.float32

            ),

            "laptops": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0, -LAPTOP_RADIUS, -(self.MAX_ADVANCE_ROBOT), -self.MAX_ROTATION] * (self.MAX_LAPTOPS if self.get_padded_observations else self.NUMBER_OF_LAPTOPS), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, LAPTOP_RADIUS, +(self.MAX_ADVANCE_ROBOT), +self.MAX_ROTATION] * (self.MAX_LAPTOPS if self.get_padded_observations else self.NUMBER_OF_LAPTOPS), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 7)*(self.MAX_LAPTOPS if self.get_padded_observations else self.NUMBER_OF_LAPTOPS),)),
                dtype=np.float32

            ),

            "tables": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0, -TABLE_RADIUS, -(self.MAX_ADVANCE_ROBOT), -self.MAX_ROTATION] * (self.MAX_TABLES if self.get_padded_observations else self.NUMBER_OF_TABLES), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, TABLE_RADIUS, +(self.MAX_ADVANCE_ROBOT), +self.MAX_ROTATION] * (self.MAX_TABLES if self.get_padded_observations else self.NUMBER_OF_TABLES), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 7)*(self.MAX_TABLES if self.get_padded_observations else self.NUMBER_OF_TABLES),)),
                dtype=np.float32

            ),

            "plants": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0, -PLANT_RADIUS, -(self.MAX_ADVANCE_ROBOT), -self.MAX_ROTATION] * (self.MAX_PLANTS if self.get_padded_observations else self.NUMBER_OF_PLANTS), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, PLANT_RADIUS, +(self.MAX_ADVANCE_ROBOT), +self.MAX_ROTATION] * (self.MAX_PLANTS if self.get_padded_observations else self.NUMBER_OF_PLANTS), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 7)*(self.MAX_PLANTS if self.get_padded_observations else self.NUMBER_OF_PLANTS),)),
                dtype=np.float32

            ),
        }

        if not self.get_padded_observations:
            total_segments = 0
            for w in self.walls:
                total_segments += w.length//self.WALL_SEGMENT_SIZE
                if w.length % self.WALL_SEGMENT_SIZE != 0: total_segments += 1
            
            d["walls"] = spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0, -self.WALL_SEGMENT_SIZE, -(self.MAX_ADVANCE_ROBOT), -self.MAX_ROTATION] * (total_segments), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, +self.WALL_SEGMENT_SIZE, +(self.MAX_ADVANCE_ROBOT), +self.MAX_ROTATION] * (total_segments), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 7)*(total_segments),)),
                dtype=np.float32
            )

        return spaces.Dict(d)


    @property
    def action_space(self): # continuous action space 
        """
        Action space contains two parameters viz linear and angular velocity. Both values lie in the range [-1, 1]. Velocities are obtained by converting the linear value to [0, self.MAX_ADVANCE_ROBOT] and the angular value to [-self.MAX_ROTATION, +self.MAX_ROTATION].
        Returns:
        gym.spaces.Box : the action space of the environment
        """
        #               adv rot
        low  = np.array([-1, -1], dtype=np.float32)
        high = np.array([+1, +1], dtype=np.float32)
        return spaces.box.Box(low, high, low.shape, np.float32)

    @property
    def done(self):
        """
        Indicates whether the episode has finished
        Returns:
        bool: True if episode has finished, and False if the episode has not finished.
        """
        return self.robot_is_done

    @property
    def transformation_matrix(self):
        """
        The transformation matrix that can convert coordinates from the global frame to the robot frame. This is calculated by inverting the transformation from the world frame to the robot frame
        
        That is,
        np.linalg.inv([[cos(theta)    -sin(theta)      h],
                       [sin(theta)     cos(theta)      k],
                       [0              0               1]]) where h, k are the coordinates of the robot in the global frame and theta is the angle of the X-axis of the robot frame with the X-axis of the global frame

        Note that the above matrix is invertible since the determinant is always 1.

        Returns:
        numpy.ndarray : the transformation matrix to convert coordinates from the world frame to the robot frame.
        """
        # check if the coordinates and orientation are not None
        assert(self.robot.x is not None and self.robot.y is not None and self.robot.orientation is not None), "Robot coordinates or orientation are None type"
        # initalizing the matrix
        tm = np.zeros((3,3), dtype=np.float32)
        # filling values as described
        tm[2,2] = 1
        tm[0,2] = self.robot.x
        tm[1,2] = self.robot.y
        tm[0,0] = tm[1,1] = np.cos(self.robot.orientation)
        tm[1,0] = np.sin(self.robot.orientation)
        tm[0,1] = -1*np.sin(self.robot.orientation)

        return np.linalg.inv(tm)

    def get_robot_frame_coordinates(self, coord):
        """
        Given coordinates in the world frame, this method returns the corresponding robot frame coordinates.
        Args:
            coord (numpy.ndarray) :  coordinate input in the world frame expressed as np.array([[x,y]]). If there are multiple coordinates, then give input as 2-D array with shape (no. of points, 2).
        Returns:
            numpy.ndarray : Coordinates in the robot frame. Shape is same as the input shape.

        """
        # converting the coordinates to homogeneous coordinates
        homogeneous_coordinates = np.c_[coord, np.ones((coord.shape[0], 1))]
        # getting the robot frame coordinates by multiplying with the transformation matrix
        coord_in_robot_frame = (self.transformation_matrix@homogeneous_coordinates.T).T
        return coord_in_robot_frame[:, 0:2]

    def get_observation(self):
        """
        Used to get the observations in the robot frame

        Returns:
            numpy.ndarray : observation as described in the observation space.
        """
        
        def observation_with_cos_sin_rather_than_angle(object): 
            """
            Returning the observation for one individual object. Also to get the sin and cos of the relative angle rather than the angle itself.
            Input:
                object (one of socnavenv.envs.utils.object.Object's subclasses) : the object of interest
            Returns:
                numpy.ndarray : the observations of the given object.
            """

            # checking the coordinates and orientation of the object are not None
            assert((object.x is not None) and (object.y is not None) and (object.orientation is not None)), f"{object.name}'s coordinates or orientation are None type"
            # initializing output array
            output = np.array([], dtype=np.float32)
            
            # object's one-hot encoding
            output = np.concatenate(
                (
                    output,
                    object.one_hot_encoding
                ),
                dtype=np.float32
            )

            # object's coordinates in the robot frame
            output = np.concatenate(
                        (
                            output,
                            self.get_robot_frame_coordinates(np.array([[object.x, object.y]])).flatten() 
                        ),
                        dtype=np.float32
                    )

            # sin and cos of the relative angle of the object
            output = np.concatenate(
                        (
                            output,
                            np.array([(np.sin(object.orientation - self.robot.orientation)), np.cos(object.orientation - self.robot.orientation)]) 
                        ),
                        dtype=np.float32
                    )

            # object's radius
            radius = 0
            if object.name == "plant":
                radius = object.radius
            elif object.name == "human":
                radius = object.width/2
            elif object.name == "table" or object.name == "laptop":
                radius = np.sqrt((object.length/2)**2 + (object.width/2)**2)
            else: raise NotImplementedError

            output = np.concatenate(
                (
                    output,
                    np.array([radius], dtype=np.float32)
                ),
                dtype=np.float32
            )


            # relative speeds for static objects
            relative_speeds = np.array([-self.robot.linear_vel, -self.robot.angular_vel], dtype=np.float32) 
            
            if object.name == "human": # the only dynamic object
                # relative linear speed
                relative_speeds[0] = np.sqrt((object.speed*np.cos(object.orientation) - self.robot.linear_vel*np.cos(self.robot.orientation))**2 + (object.speed*np.sin(object.orientation) - self.robot.linear_vel*np.sin(self.robot.orientation))**2) 
            
            output = np.concatenate(
                        (
                            output,
                            relative_speeds
                        ),
                        dtype=np.float32
                    )
            return output.flatten()

        def segment_wall(wall:Wall, size:float):
            centers = []
            lengths = []

            left_x = wall.x - wall.length/2 * np.cos(wall.orientation)
            left_y = wall.y - wall.length/2 * np.sin(wall.orientation)

            right_x = wall.x + wall.length/2 * np.cos(wall.orientation)
            right_y = wall.y + wall.length/2 * np.sin(wall.orientation)

            segment_x = left_x + np.cos(wall.orientation)*(size/2)
            segment_y = left_y + np.sin(wall.orientation)*(size/2)

            for i in range(int(wall.length//size)):
                centers.append((segment_x, segment_y))
                lengths.append(size)
                segment_x += np.cos(wall.orientation)*size
                segment_y += np.sin(wall.orientation)*size

            if(wall.length % size != 0):
                length = wall.length % size
                centers.append((right_x - np.cos(wall.orientation)*length/2, right_y - np.sin(wall.orientation)*length/2))
                lengths.append(length)
            
            return centers, lengths

        def get_wall_observations(size:int):
            obs = np.array([], dtype=np.float32)
            for w in self.walls:
                c, l = segment_wall(w, size)    
                for center, length in zip(c, l):
                    obs = np.concatenate((obs, w.one_hot_encoding))
                    obs = np.concatenate((obs, self.get_robot_frame_coordinates(np.array([[center[0], center[1]]])).flatten()))
                    obs = np.concatenate((obs, np.array([(np.sin(w.orientation - self.robot.orientation)), np.cos(w.orientation - self.robot.orientation)])))
                    obs = np.concatenate((obs, np.array([length/2])))
                    relative_speeds = np.array([-self.robot.linear_vel, -self.robot.angular_vel], dtype=np.float32)
                    obs = np.concatenate((obs, relative_speeds))
                    obs = obs.flatten().astype(np.float32)
            return obs  

        # the observations will go inside this dictionary
        d = {}
        
        # goal coordinates in the robot frame
        goal_in_robot_frame = self.get_robot_frame_coordinates(np.array([[self.robot.goal_x, self.robot.goal_y]], dtype=np.float32))
        # converting into the required shape
        goal_obs = goal_in_robot_frame.flatten()

        # concatenating with the robot's one-hot-encoding
        goal_obs = np.concatenate((self.robot.one_hot_encoding, goal_obs), dtype=np.float32)
        # placing it in a dictionary
        d["goal"] = goal_obs
        
        # getting the observations of humans
        human_obs = np.array([], dtype=np.float32)
        for human in self.humans:
            obs = observation_with_cos_sin_rather_than_angle(human)
            human_obs = np.concatenate((human_obs, obs), dtype=np.float32)
       
        if self.get_padded_observations:
            # padding with zeros
            human_obs = np.concatenate((human_obs, np.zeros(self.observation_space["humans"].shape[0] -human_obs.shape[0])), dtype=np.float32)
        
        # inserting in the dictionary
        d["humans"] = human_obs

    
        # getting the observations of laptops
        laptop_obs = np.array([], dtype=np.float32)
        for laptop in self.laptops:
            obs = observation_with_cos_sin_rather_than_angle(laptop)
            laptop_obs = np.concatenate((laptop_obs, obs), dtype=np.float32)
       
        if self.get_padded_observations:
            # padding with zeros
            laptop_obs = np.concatenate((laptop_obs, np.zeros(self.observation_space["laptops"].shape[0] -laptop_obs.shape[0])), dtype=np.float32)
        
        # inserting in the dictionary
        d["laptops"] = laptop_obs
    

        # getting the observations of tables
        table_obs = np.array([], dtype=np.float32)
        for table in self.tables:
            obs = observation_with_cos_sin_rather_than_angle(table)
            table_obs = np.concatenate((table_obs, obs), dtype=np.float32)

        if self.get_padded_observations:
            # padding with zeros
            table_obs = np.concatenate((table_obs, np.zeros(self.observation_space["tables"].shape[0] -table_obs.shape[0])), dtype=np.float32)
        
        # inserting in the dictionary
        d["tables"] = table_obs


        # getting the observations of plants
        plant_obs = np.array([], dtype=np.float32)
        for plant in self.plants:
            obs = observation_with_cos_sin_rather_than_angle(plant)
            plant_obs = np.concatenate((plant_obs, obs), dtype=np.float32)

        if self.get_padded_observations:
            # padding with zeros
            plant_obs = np.concatenate((plant_obs, np.zeros(self.observation_space["plants"].shape[0] -plant_obs.shape[0])), dtype=np.float32)
        
        # inserting in the dictionary
        d["plants"] = plant_obs

        # inserting wall observations to the dictionary
        if not self.get_padded_observations:
            d["walls"] = get_wall_observations(self.WALL_SEGMENT_SIZE)

        return d

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        """
        Calculating environment forces  Reference : https://github.com/openai/multiagent-particle-envs/blob/master/multiagent/core.py 
        """
       
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        
        # compute actual distance between entities
        delta_pos = np.array([entity_a.x - entity_b.x, entity_a.y - entity_b.y], dtype=np.float32) 
        # minimum allowable distance
        dist = np.sqrt(np.sum(np.square(delta_pos)))

        # calculating the radius based on the entitiy
        if entity_a.name == "plant" or entity_a.name == "robot": # circular shaped
            radius_a = entity_a.radius
        
        # width was assumed as the diameter of the human
        elif entity_a.name == "human": 
            radius_a = entity_a.width/2

        # initialized to 0. Walls are separately handled below
        elif entity_a.name == "wall":  
            radius_a = 0
        
        # approximating the rectangular objects with a circle that circumscribes it
        elif  entity_a.name == "table" or entity_a.name == "laptop":
            radius_a = np.sqrt((entity_a.length/2)**2 + (entity_a.width/2)**2)

        else: raise NotImplementedError

        # similarly calculating for entity b
        if entity_b.name == "plant" or entity_b.name == "robot":
            radius_b = entity_b.radius
        
        elif entity_b.name == "human":
            radius_b = entity_b.width/2
        
        elif entity_b.name == "wall":
            radius_b = 0
        
        elif  entity_b.name == "table" or entity_b.name == "laptop":
            radius_b = np.sqrt((entity_b.length/2)**2 + (entity_b.width/2)**2)
        
        else: raise NotImplementedError
        
        # if one of the entities is a wall, the center is taken to be the reflection of the point in the wall, and radius same as the other entity
        if entity_a.name == "wall":
            if entity_a.orientation == np.pi/2 or entity_a.orientation == -np.pi/2:
                # taking reflection about the striaght line parallel to y axis
                center_x = 2*entity_a.x - entity_b.x  + ((entity_a.thickness) if entity_b.x >= entity_a.x else (-entity_a.thickness))
                center_y = entity_b.y
                delta_pos = np.array([center_x - entity_b.x, center_y - entity_b.y], dtype=np.float32) 
            
            elif entity_a.orientation == 0 or entity_a.orientation == np.pi:
                # taking reflection about a striaght line parallel to the x axis
                center_x = entity_b.x
                center_y = 2*entity_a.y - entity_b.y + ((entity_a.thickness) if entity_b.y >= entity_a.y else (-entity_a.thickness))
                delta_pos = np.array([center_x - entity_b.x, center_y - entity_b.y], dtype=np.float32) 

            else : raise NotImplementedError
            # setting the radius of the wall to be the radius of the entity. This is done because the wall's center was assumed to be the reflection of the other entity's center, so now to collide with the wall, it should collide with a circle of the same size.
            radius_a = radius_b
            dist = np.sqrt(np.sum(np.square(delta_pos)))

        elif entity_b.name == "wall":
            if entity_b.orientation == np.pi/2 or entity_b.orientation == -np.pi/2:
                center_x = 2*entity_b.x - entity_a.x  + ((entity_b.thickness) if entity_a.x >= entity_b.x else (-entity_b.thickness))
                center_y = entity_a.y
                delta_pos = np.array([entity_a.x - center_x, entity_a.y - center_y], dtype=np.float32) 
            
            elif entity_b.orientation == 0 or entity_b.orientation == np.pi:
                center_x = entity_a.x
                center_y = 2*entity_b.y - entity_a.y + ((entity_b.thickness) if entity_a.y >= entity_b.y else (-entity_b.thickness))
                delta_pos = np.array([entity_a.x - center_x, entity_a.y - center_y], dtype=np.float32) 

            else : raise NotImplementedError

            radius_b = radius_a
            dist = np.sqrt(np.sum(np.square(delta_pos)))

        # minimum distance that is possible between two entities
        dist_min = radius_a + radius_b
        
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if not entity_a.is_static else None  # forces are applied only to dynamic objects
        force_b = -force if not entity_b.is_static else None  # forces are applied only to dynamic objects
        return [force_a, force_b]

       
    def step(self, action_pre, update=True):
        """
        Computes a step in the current episode given the action.
        Input:
            action_pre (numpy.ndarray or list) : An action that lies in the action space
        Returns:
            observation (numpy.ndarray) : the observation from the current action
            reward (float) : reward received on the current action
            done (bool) : whether the episode has finished or not
            info (dict) : additional information
        """
        # for converting the action to the velocity
        def process_action(act):
            action = act.astype(np.float32)
            action[0] = (float(action[0]+1.0)/2.0)*self.MAX_ADVANCE_ROBOT   # [-1, +1] --> [0, self.MAX_ADVANCE_ROBOT]
            action[1] = (float(action[1]+0.0)/1.0)*self.MAX_ROTATION  # [-1, +1] --> [-self.MAX_ROTATION, +self.MAX_ROTATION]
            if action[0] < 0:               # Advance must be negative
                action[0] *= -1
            if action[0] > self.MAX_ADVANCE_ROBOT:     # Advance must be less or equal self.MAX_ADVANCE_ROBOT
                action[0] = self.MAX_ADVANCE_ROBOT
            if action[1]   < -self.MAX_ROTATION:   # Rotation must be higher than -self.MAX_ROTATION
                action[1] =  -self.MAX_ROTATION
            elif action[1] > +self.MAX_ROTATION:  # Rotation must be lower than +self.MAX_ROTATION
                action[1] =  +self.MAX_ROTATION

            return action

        # if action is a list, converting it to numpy.ndarray
        if(type(action_pre) == list):
            action_pre = np.array(action_pre, dtype=np.float32)

        # call error if the environment wasn't reset after the episode ended
        if self.robot_is_done:
            raise Exception('step call within a finished episode!')
    
        # calculating the velocity from action
        action = process_action(action_pre)
        # setting the robot's linear and angular velocity
        self.robot.linear_vel = action[0]
        self.robot.angular_vel = action[1]

        if update:
            # update robot
            self.robot.update(TIMESTEP)

            # update humans
            for human in self.humans:
                human.update(TIMESTEP)

        # getting observations
        observation = self.get_observation()

        # computing rewards and done 
        reward = self.compute_reward_and_ticks()
        done = self.robot_is_done
        info = {}

        # if done: sys.exit(0)
        self.cumulative_reward += reward

        # providing debugging information
        if DEBUG > 0 and self.ticks%50==0:
            self.render()
        elif DEBUG > 1:
            self.render()

        if DEBUG > 0 and self.robot_is_done:
            print(f'cumulative reward: {self.cumulative_reward}')

        return observation, reward, done, info


    def compute_reward_and_ticks(self):
        """
        Function to compute the reward and also calculate if the episode has finished
        """
        self.ticks += 1

        # calculate the distance to the goal
        distance_to_goal = np.sqrt((self.robot.goal_x - self.robot.x)**2 + (self.robot.goal_y - self.robot.y)**2)

        # check for object-robot collisions
        collision = False

        for object in self.humans + self.plants + self.walls + self.tables + self.laptops :
            if(self.robot.collides(object)): 
                collision = True
                """
                Updating the robot velocity to continue with the episode
                """
                if object.name == "human":
                    break

                [fa, fb] = self.get_collision_force(self.robot, object)
                if (fa is not None):
                    entity_vel = (fa / self.robot.mass) * TIMESTEP
                    self.robot.update_orientation(entity_vel[0], entity_vel[1])
                break
        
        for human in self.humans:    
            for object in self.plants + self.tables + self.laptops:
                if human.collides(object):
                    [fi, fj] = self.get_collision_force(human, object)

                    if(fi is not None):
                        entity_vel = (fi / human.mass) * TIMESTEP     
                        if human.reroute_steps == 0:       
                            human.reroute(entity_vel[0], entity_vel[1], 2)
        
        for i in range(len(self.humans)):
            for j in range(i+1, len(self.humans)):
                if self.humans[i].collides(self.humans[j]):
                    [fi, fj] = self.get_collision_force(self.humans[i], self.humans[j])

                    if(fi is not None):
                        entity_vel = (fi / self.humans[i].mass) * TIMESTEP  
                        if self.humans[i].reroute_steps == 0:       
                            self.humans[i].reroute(entity_vel[0], entity_vel[1], 1)

                    if(fj is not None):
                        entity_vel = (fj / self.humans[j].mass) * TIMESTEP                 
                        if self.humans[j].reroute_steps == 0:       
                            self.humans[j].reroute(entity_vel[0], entity_vel[1], 1)

        
        for human in self.humans:
            for wall in self.walls:
                if human.collides(wall):
                    [f, fj] = self.get_collision_force(human, wall)

                    if(f is not None):
                        entity_vel = (f / human.mass) * TIMESTEP                 
                        human.update_orientation(entity_vel[0], entity_vel[1])
                

        for human in self.humans:
            if human.collides(self.robot):
                [fi, fj] = self.get_collision_force(human, self.robot)
                if(fi is not None):
                    entity_vel = (fi / human.mass) * TIMESTEP    
                    if human.reroute_steps == 0:       
                        human.reroute(entity_vel[0], entity_vel[1], 2)


        # calculate the reward and update is_done
        if self.MAP_X/2 < self.robot.x or self.robot.x < -self.MAP_X/2 or self.MAP_Y/2 < self.robot.y or self.robot.y < -self.MAP_Y/2:
            self.robot_is_done = True
            reward = OUTOFMAP_REWARD
        elif distance_to_goal < GOAL_THRESHOLD:
            self.robot_is_done = True
            reward = REACH_REWARD
        elif collision is True:
            self.robot_is_done = True
            reward = COLLISION_REWARD
        elif self.ticks > MAX_TICKS:
            self.robot_is_done = True
            reward = MAXTICKS_REWARD
        else:
            self.robot_is_done = False
            reward = -distance_to_goal/DISTANCE_REWARD_DIVISOR + ALIVE_REWARD

        return reward

    def reset(self) :
        """
        Resets the environment
        """
        self.cumulative_reward = 0
        # randomly select the shape
        # self.shape = random.choice(["rectangle", "square", "L"])
        self.shape = "square"

        # randomly initialize the parameters 
        self.randomize_params()

        HALF_SIZE_X = self.MAP_X/2. - self.MARGIN
        HALF_SIZE_Y = self.MAP_Y/2. - self.MARGIN
        
        # to keep track of the current objects
        self.objects = []
        self.laptops = []
        self.walls = []
        self.humans = []
        self.plants = []
        self.tables = []

        if self.shape == "L":
            # keep the direction of this as well
            location = np.random.randint(0,4)
            
            if location == 0:
                # top right
                l = Laptop(
                    x=self.MAP_X/2.0- self.L_X/2.0,
                    y=self.MAP_Y/2.0 - self.L_Y/2.0,
                    width=self.L_Y,
                    length=self.L_X,
                    theta=0
                )
                # adding walls
                w_l1 = Wall(x=self.MAP_X/2 -self.L_X, y=self.MAP_Y/2 -self.L_Y/2, theta=np.pi/2, length=self.L_Y, thickness=WALL_THICKNESS)
                w_l2 = Wall(x=self.MAP_X/2 -self.L_X/2, y=self.MAP_Y/2 -self.L_Y, theta=0, length=self.L_X, thickness=WALL_THICKNESS)
                w_l3 = Wall(x=self.MAP_X/2-(WALL_THICKNESS/2), y=-self.L_Y/2, theta=np.pi/2, length=self.MAP_Y-self.L_Y, thickness=WALL_THICKNESS)
                w_l4 = Wall(x=0, y=-self.MAP_Y/2 + (WALL_THICKNESS/2), theta=0, length=self.MAP_X, thickness=WALL_THICKNESS)
                w_l5 = Wall(x=-self.MAP_X/2 + (WALL_THICKNESS/2), y=0, theta=np.pi/2, length=self.MAP_Y, thickness=WALL_THICKNESS)
                w_l6 = Wall(x=-self.L_X/2, y=self.MAP_Y/2-(WALL_THICKNESS/2), theta=0, length=self.MAP_X-self.L_X, thickness=WALL_THICKNESS)

            elif location == 1:
                # top left
                l = Laptop(
                    x=-self.MAP_X/2.0 + self.L_X/2.0,
                    y=self.MAP_Y/2.0 - self.L_Y/2.0,
                    width=self.L_Y,
                    length=self.L_X,
                    theta=0
                )
                # adding walls
                w_l1 = Wall(x=-self.MAP_X/2 + self.L_X, y=self.MAP_Y/2 -self.L_Y/2, theta=np.pi/2, length=self.L_Y, thickness=WALL_THICKNESS)
                w_l2 = Wall(x=-self.MAP_X/2 +self.L_X/2, y=self.MAP_Y/2 -self.L_Y, theta=0, length=self.L_X, thickness=WALL_THICKNESS)
                w_l3 = Wall(x=-self.MAP_X/2+(WALL_THICKNESS/2), y=-self.L_Y/2, theta=np.pi/2, length=self.MAP_Y-self.L_Y, thickness=WALL_THICKNESS)
                w_l4 = Wall(x=0, y=-self.MAP_Y/2 + (WALL_THICKNESS/2), theta=0, length=self.MAP_X, thickness=WALL_THICKNESS)
                w_l5 = Wall(x=self.MAP_X/2-(WALL_THICKNESS/2), y=0, theta=np.pi/2, length=self.MAP_Y, thickness=WALL_THICKNESS)
                w_l6 = Wall(x=self.L_X/2, y=self.MAP_Y/2-(WALL_THICKNESS/2), theta=0, length=self.MAP_X-self.L_X, thickness=WALL_THICKNESS)
            
            elif location == 2:
                # bottom right
                l = Laptop(
                    x=self.MAP_X/2.0 - self.L_X/2.0,
                    y=-self.MAP_Y/2.0 + self.L_Y/2.0,
                    width=self.L_Y,
                    length=self.L_X,
                    theta=0
                )
                # adding walls
                w_l1 = Wall(x=self.MAP_X/2 - self.L_X, y=-self.MAP_Y/2 + self.L_Y/2, theta=np.pi/2,length=self.L_Y, thickness=WALL_THICKNESS)
                w_l2 = Wall(x=self.MAP_X/2 - self.L_X/2, y=-self.MAP_Y/2 +self.L_Y, theta=0, length=self.L_X, thickness=WALL_THICKNESS)
                w_l3 = Wall(x=self.MAP_X/2-(WALL_THICKNESS/2), y=self.L_Y/2, theta=np.pi/2, length=self.MAP_Y-self.L_Y, thickness=WALL_THICKNESS)
                w_l4 = Wall(x=0, y=self.MAP_Y/2-(WALL_THICKNESS/2), theta=0, length=self.MAP_X, thickness=WALL_THICKNESS)
                w_l5 = Wall(x=-self.MAP_X/2+(WALL_THICKNESS/2), y=0, theta=np.pi/2, length=self.MAP_Y, thickness=WALL_THICKNESS)
                w_l6 = Wall(x=-self.L_X/2, y=-self.MAP_Y/2+(WALL_THICKNESS/2), theta=0, length=self.MAP_X-self.L_X, thickness=WALL_THICKNESS)

            elif location == 3:
                # bottom left
                l = Laptop(
                    x=-self.MAP_X/2.0 + self.L_X/2.0,
                    y=-self.MAP_Y/2.0 + self.L_Y/2.0,
                    width=self.L_Y,
                    length=self.L_X,
                    theta=0
                )
                # adding walls
                w_l1 = Wall(x= -self.MAP_X/2 +self.L_X, y= -self.MAP_Y/2 + self.L_Y/2, theta=np.pi/2, length=self.L_Y, thickness=WALL_THICKNESS)
                w_l2 = Wall(x=-self.MAP_X/2 + self.L_X/2, y=-self.MAP_Y/2 + self.L_Y, theta=0, length=self.L_X, thickness=WALL_THICKNESS)
                w_l3 = Wall(x=-self.MAP_X/2+(WALL_THICKNESS/2), y=self.L_Y/2, theta=np.pi/2, length=self.MAP_Y-self.L_Y, thickness=WALL_THICKNESS)
                w_l4 = Wall(x=0, y=self.MAP_Y/2-(WALL_THICKNESS/2), theta=0, length=self.MAP_X, thickness=WALL_THICKNESS)
                w_l5 = Wall(x=self.MAP_X/2-(WALL_THICKNESS/2), y=0, theta=np.pi/2, length=self.MAP_Y, thickness=WALL_THICKNESS)
                w_l6 = Wall(x=self.L_X/2, y=-self.MAP_Y/2+(WALL_THICKNESS/2), theta=0, length=self.MAP_X-self.L_X, thickness=WALL_THICKNESS)

            self.objects.append(l)
            self.walls.append(w_l1)
            self.walls.append(w_l2)
            self.walls.append(w_l3)
            self.walls.append(w_l4)
            self.walls.append(w_l5)
            self.walls.append(w_l6)
            self.objects.append(w_l1)
            self.objects.append(w_l2)
            self.objects.append(w_l3)
            self.objects.append(w_l4)
            self.objects.append(w_l5)
            self.objects.append(w_l6)

        # walls (hardcoded to be at the boundaries of the environment)
        else:
            w1 = Wall(0, self.MAP_Y/2-WALL_THICKNESS/2, 0, self.MAP_X, WALL_THICKNESS)
            w2 = Wall(self.MAP_X/2-WALL_THICKNESS/2, 0, np.pi/2, self.MAP_Y, WALL_THICKNESS)
            w3 = Wall(0, -self.MAP_Y/2+WALL_THICKNESS/2, 0, self.MAP_X, WALL_THICKNESS)
            w4 = Wall(-self.MAP_X/2+WALL_THICKNESS/2, 0, np.pi/2, self.MAP_Y, WALL_THICKNESS)
            self.walls.append(w1)
            self.walls.append(w2)
            self.walls.append(w3)
            self.walls.append(w4)
            self.objects.append(w1)
            self.objects.append(w2)
            self.objects.append(w3)
            self.objects.append(w4)


        # robot
        while True:
            robot = Robot(
                x = random.uniform(-HALF_SIZE_X, HALF_SIZE_X),
                y = random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y),
                theta = random.uniform(-np.pi, np.pi),
                radius = ROBOT_RADIUS,
                goal_x = random.uniform(-HALF_SIZE_X, HALF_SIZE_X),
                goal_y = random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y)
            )
            collides = False
            for obj in self.objects: # check if spawned object collides with any of the exisiting objects
                if(robot.collides(obj)):
                    collides = True
                    break

            if collides:
                del robot
            else:
                self.robot = robot
                self.objects.append(self.robot)
                break

        # setting the goal
        # adding a plant obstacle as the goal so that the other obstacles that are created do not collide with the goal 
        while True:
            plant = Plant(
                x = random.uniform(-HALF_SIZE_X, HALF_SIZE_X),
                y = random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y),
                radius=GOAL_RADIUS
            )
            collides = False
            for obj in self.objects: # check if spawned object collides with any of the exisiting objects. It will not be rendered as a plant.
                if(plant.collides(obj)):
                    collides = True
                    break

            if collides:
                del plant
            else:
                self.robot.goal_x = plant.x
                self.robot.goal_y = plant.y
                self.objects.append(plant)
                break

        # humans
        for i in range(self.NUMBER_OF_HUMANS): # spawn specified number of humans
            while True: # comes out of loop only when spawned object collides with none of current objects
                x = random.uniform(-HALF_SIZE_X, HALF_SIZE_X)
                y = random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y)
                        
                human = Human(
                    x=x,
                    y=y,
                    theta=random.uniform(-np.pi, np.pi) ,
                    width=HUMAN_DIAMETER,
                    speed=random.uniform(0.0, self.MAX_ADVANCE_HUMAN)
                )

                collides = False
                for obj in self.objects: # check if spawned object collides with any of the exisiting objects
                    if(human.collides(obj)):
                        collides = True
                        break

                if collides:
                    del human
                else:
                    self.humans.append(human)
                    self.objects.append(human)
                    break
        
        # plants
        for i in range(self.NUMBER_OF_PLANTS): # spawn specified number of plants
            while True: # comes out of loop only when spawned object collides with none of current objects
                x = random.uniform(-HALF_SIZE_X, HALF_SIZE_X)
                y = random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y)
                        
                plant = Plant(
                    x=x,
                    y=y,
                    radius=PLANT_RADIUS
                )

                collides = False
                for obj in self.objects:
                    if(plant.collides(obj)):
                        collides = True
                        break

                if collides:
                    del plant
                else:
                    self.plants.append(plant)
                    self.objects.append(plant)
                    break

        # tables
        for i in range(self.NUMBER_OF_TABLES): # spawn specified number of tables
            while True: # comes out of loop only when spawned object collides with none of current objects
                x = random.uniform(-HALF_SIZE_X, HALF_SIZE_X)
                y = random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y)
                        
                table = Table(
                    x=x,
                    y=y,
                    theta=random.uniform(-np.pi, np.pi),
                    width=TABLE_WIDTH,
                    length=TABLE_LENGTH
                )

                collides = False
                for obj in self.objects:
                    if(table.collides(obj)):
                        collides = True
                        break

                if collides:
                    del table
                else:
                    self.tables.append(table)
                    self.objects.append(table)
                    break

        # laptops
        if(len(self.tables) == 0):
            "print: No tables found, placing laptops on the floor!"
            for i in range(self.NUMBER_OF_LAPTOPS): # spawn specified number of laptops
                while True: # comes out of loop only when spawned object collides with none of current objects
                    x = random.uniform(-HALF_SIZE_X, HALF_SIZE_X)
                    y = random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y)
                            
                    laptop = Laptop(
                        x=x,
                        y=y,
                        theta=random.uniform(-np.pi, np.pi),
                        width=LAPTOP_WIDTH,
                        length=LAPTOP_LENGTH
                    )

                    collides = False
                    for obj in self.objects:
                        if(laptop.collides(obj)):
                            collides = True
                            break

                    if collides:
                        del laptop
                    else:
                        self.laptops.append(laptop)
                        self.objects.append(laptop)
                        break
        
        else:
            for _ in range(self.NUMBER_OF_LAPTOPS): # placing laptops on tables
                i = random.randint(0, len(self.tables)-1)
                table = self.tables[i]
                
                while True: # comes out of loop only when spawned object collides with none of current objects
                    x, y = uniform_circular_sampler(table.x, table.y, min(table.width/2, table.length/2)) # sampling the center of the laptop on a circle with center as the center of the table and radius as min(length/2, breadth/2)
                    laptop = Laptop(
                        x=x,
                        y=y,
                        theta=random.uniform(-np.pi, np.pi),
                        width=LAPTOP_WIDTH,
                        length=LAPTOP_LENGTH
                    )

                    collides = False
                    for obj in self.laptops: # it should not collide with any laptop on the table
                        if(laptop.collides(obj)):
                            collides = True
                            break

                    if collides:
                        del laptop
                    else:
                        self.laptops.append(laptop)
                        self.objects.append(laptop)
                        break
        
        self.robot_is_done = False
        self.ticks = 0

        # all entities in the environment
        self.entities = self.humans + self.tables + self.laptops + self.plants + self.walls
        self.entities.append(self.robot)
        
        return self.get_observation()

    def render(self, mode="human"):
        """
        Visualizing the environment
        """

        if not self.window_initialised:
            cv2.namedWindow("world", cv2.WINDOW_NORMAL) 
            cv2.resizeWindow("world", int(RESOLUTION_VIEW), int(RESOLUTION_VIEW))
            self.window_initialised = True
        
        self.world_image = (np.ones((int(self.RESOLUTION_Y),int(self.RESOLUTION_X),3))*255).astype(np.uint8)

        for wall in self.walls:
            wall.draw(self.world_image, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)

        for table in self.tables:
            table.draw(self.world_image, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)

        for laptop in self.laptops:
            laptop.draw(self.world_image, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)
        
        for plant in self.plants:
            plant.draw(self.world_image, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)

        cv2.circle(self.world_image, (w2px(self.robot.goal_x, self.PIXEL_TO_WORLD_X, self.MAP_X), w2py(self.robot.goal_y, self.PIXEL_TO_WORLD_Y, self.MAP_Y)), int(w2px(self.robot.x + GOAL_RADIUS, self.PIXEL_TO_WORLD_X, self.MAP_X) - w2px(self.robot.x, self.PIXEL_TO_WORLD_X, self.MAP_X)), (0, 255, 0), 2)
        
        for human in self.humans:
            human.draw(self.world_image, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)
        
        self.robot.draw(self.world_image, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)

        cv2.imshow("world", self.world_image)
        k = cv2.waitKey(MILLISECONDS)
        if k%255 == 27:
            sys.exit(0)

    def close(self):
        pass
