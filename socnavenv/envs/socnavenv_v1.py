import os
import random
import sys
import time
from math import atan2
from typing import List
import copy

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import rvo2
import torch
import yaml
from gym import spaces

from socnavenv.envs.utils.human import Human
from socnavenv.envs.utils.human_human import Human_Human_Interaction
from socnavenv.envs.utils.human_laptop import Human_Laptop_Interaction
from socnavenv.envs.utils.laptop import Laptop
from socnavenv.envs.utils.object import Object
from socnavenv.envs.utils.plant import Plant
from socnavenv.envs.utils.robot import Robot
from socnavenv.envs.utils.table import Table
from socnavenv.envs.utils.utils import (get_coordinates_of_rotated_rectangle,
                                        get_nearest_point_from_rectangle,
                                        get_square_around_circle,
                                        point_to_segment_dist, w2px, w2py)
from socnavenv.envs.utils.wall import Wall

sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/utils/sngnnv2")
from socnavenv.envs.utils.sngnnv2.socnav import SocNavDataset
from socnavenv.envs.utils.sngnnv2.socnav_V2_API import Human as otherHuman
from socnavenv.envs.utils.sngnnv2.socnav_V2_API import Object as otherObject
from socnavenv.envs.utils.sngnnv2.socnav_V2_API import SNScenario, SocNavAPI

DEBUG = 0
if 'debug' in sys.argv or "debug=2" in sys.argv:
    DEBUG = 2
elif "debug=1" in sys.argv:
    DEBUG = 1

class SocNavEnv_v1(gym.Env):
    """
    Class for the environment
    """
    metadata = {}
    
    # rendering params
    RESOLUTION_VIEW = None
    MILLISECONDS = None

    # episode params
    EPISODE_LENGTH = None
    TIMESTEP = None

    # rewards
    USE_SNGNN = None
    USE_DISTANCE_TO_GOAL = None
    REACH_REWARD = None
    OUTOFMAP_REWARD = None
    MAX_STEPS_REWARD = None
    ALIVE_REWARD = None
    COLLISION_REWARD = None
    DISTANCE_REWARD_SCALER = None

    # robot params
    ROBOT_RADIUS = None
    GOAL_RADIUS = None

    # human params
    HUMAN_DIAMETER = None
    HUMAN_GOAL_RADIUS = None
    HUMAN_POLICY=None

    # laptop params
    LAPTOP_WIDTH=None
    LAPTOP_LENGTH=None
    
    # plant params
    PLANT_RADIUS =None

    # table params
    TABLE_WIDTH = None
    TABLE_LENGTH = None

    # wall params
    WALL_THICKNESS = None

    # human-human interaction params
    INTERACTION_RADIUS = None
    INTERACTION_GOAL_RADIUS = None
    INTERACTION_NOISE_VARIANCE = None

    # human-laptop interaction params
    HUMAN_LAPTOP_DISTANCE = None

    def __init__(self) -> None:
        super().__init__()
        self.window_initialised = False
        self.has_configured = False
        # the number of steps taken in the current episode
        self.ticks = 0  
        # humans in the environment
        self.humans:List[Human] = [] 
        # laptops in the environment
        self.laptops:List[Laptop] = [] 
        # walls in the environment
        self.walls:List[Wall] = []  
        # plants in the environment
        self.plants:List[Plant] = []  
        # tables in the environment
        self.tables:List[Table] = []  
        # dynamic interactions
        self.moving_interactions:List[Human_Human_Interaction] = []
        # static interactions
        self.static_interactions:List[Human_Human_Interaction] = []
        # human-laptop-interactions
        self.h_l_interactions:List[Human_Laptop_Interaction] = []

        # all entities in the environment
        self.entities = None 
        
        # robot
        self.robot:Robot = None

        # environment parameters
        self.MARGIN = None
        self.MAX_ADVANCE_HUMAN = None
        self.MAX_ADVANCE_ROBOT = None
        self.MAX_ROTATION = None
        self.SPEED_THRESHOLD = None
        
        # wall segment size
        self.WALL_SEGMENT_SIZE = None


        # defining the bounds of the number of entities
        self.MIN_HUMANS = None
        self.MAX_HUMANS = None 
        self.MIN_TABLES = None
        self.MAX_TABLES = None
        self.MIN_PLANTS = None
        self.MAX_PLANTS = None
        self.MIN_LAPTOPS = None
        self.MAX_LAPTOPS = None
        self.MIN_H_H_DYNAMIC_INTERACTIONS = None
        self.MAX_H_H_DYNAMIC_INTERACTIONS = None
        self.MIN_H_H_STATIC_INTERACTIONS = None
        self.MAX_H_H_STATIC_INTERACTIONS = None
        self.MIN_HUMAN_IN_H_H_INTERACTIONS = None
        self.MAX_HUMAN_IN_H_H_INTERACTIONS = None
        self.MIN_H_L_INTERACTIONS = None
        self.MAX_H_L_INTERACTIONS = None
        self.MIN_MAP_X = None
        self.MAX_MAP_X = None
        self.MIN_MAP_Y = None
        self.MIN_MAP_Y = None

        # flag parameter that controls whether padded observations will be returned or not
        self.get_padded_observations = None

        # to check if the episode has finished
        self.robot_is_done = True
        
        # for rendering the world to an OpenCV image
        self.world_image = None
        
        # parameters for integrating multiagent particle environment's forces

        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

        # shape of the environment
        self.set_shape = None
        self.shape = None

        # rewards
        self.prev_distance = None


    def configure(self, config_path):
        """
        To read from config file to set env parameters
        
        Args:
            config_path(str): path to config file
        """
        # loading config file
        with open(config_path, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)

        self.has_configured = True

        # resolution view
        self.RESOLUTION_VIEW = config["rendering"]["resolution_view"]
        assert(self.RESOLUTION_VIEW > 0), "resolution view should be greater than 0"
        
        # milliseconds
        self.MILLISECONDS = config["rendering"]["milliseconds"]
        assert(self.MILLISECONDS > 0), "milliseconds should be greater than zero"

        # episode parameters
        self.EPISODE_LENGTH = config["episode"]["episode_length"]
        assert(self.EPISODE_LENGTH > 0), "episode length should be greater than 0"
        self.TIMESTEP = config["episode"]["time_step"]

        # rewards
        self.USE_SNGNN = config["rewards"]["use_sngnn"]
        if self.USE_SNGNN: 
            self.sngnn = SocNavAPI(device= ('cuda' if torch.cuda.is_available() else 'cpu'), params_dir=(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", "sngnnv2", "example_model")))

        self.USE_DISTANCE_TO_GOAL = config["rewards"]["use_distance_to_goal"]
        self.DISTANCE_REWARD_SCALER = config["rewards"]["distance_reward_scaler"]

        self.REACH_REWARD = config["rewards"]["reach_reward"]
        self.OUTOFMAP_REWARD = config["rewards"]["out_of_map_reward"]
        self.MAX_STEPS_REWARD = config["rewards"]["max_steps_reward"]
        self.ALIVE_REWARD = config["rewards"]["alive_reward"]
        self.COLLISION_REWARD = config["rewards"]["collision_reward"]
        self.DISCOMFORT_DISTANCE = config["rewards"]["discomfort_dist"]
        self.DISCOMFORT_PENALTY_FACTOR = config["rewards"]["discomfort_penalty_factor"]

        # robot
        self.ROBOT_RADIUS = config["robot"]["robot_radius"]
        self.GOAL_RADIUS = config["robot"]["goal_radius"]
        assert(self.ROBOT_RADIUS > 0 and self.GOAL_RADIUS > 0), "robot parameters in config file should be greater than 0"
        self.GOAL_THRESHOLD = self.ROBOT_RADIUS + self.GOAL_RADIUS

        # human
        self.HUMAN_DIAMETER = config["human"]["human_diameter"]
        self.HUMAN_GOAL_RADIUS = config["human"]["human_goal_radius"]
        self.HUMAN_POLICY = config["human"]["human_policy"]
        assert(self.HUMAN_POLICY=="random" or self.HUMAN_POLICY == "orca" or self.HUMAN_POLICY == "sfm"), "human_policy should be \"random\", or \"orca\" or \"sfm\""

        # laptop
        self.LAPTOP_WIDTH = config["laptop"]["laptop_width"]
        self.LAPTOP_LENGTH = config["laptop"]["laptop_length"]
        self.LAPTOP_RADIUS = np.sqrt((self.LAPTOP_LENGTH/2)**2 + (self.LAPTOP_WIDTH/2)**2)

        # plant
        self.PLANT_RADIUS = config["plant"]["plant_radius"]

        # table
        self.TABLE_WIDTH = config["table"]["table_width"]
        self.TABLE_LENGTH = config["table"]["table_length"]
        self.TABLE_RADIUS = np.sqrt((self.TABLE_LENGTH/2)**2 + (self.TABLE_WIDTH/2)**2)

        # wall
        self.WALL_THICKNESS = config["wall"]["wall_thickness"]

        # human-human-interaction
        self.INTERACTION_RADIUS = config["human-human-interaction"]["interaction_radius"]
        self.INTERACTION_GOAL_RADIUS = config["human-human-interaction"]["interaction_goal_radius"]
        self.INTERACTION_NOISE_VARIANCE = config["human-human-interaction"]["noise_variance"]

        # human-laptop-interaction
        self.HUMAN_LAPTOP_DISTANCE = config["human-laptop-interaction"]["human_laptop_distance"]
        
        # env
        self.MARGIN = config["env"]["margin"]
        self.MAX_ADVANCE_HUMAN = config["env"]["max_advance_human"]
        self.MAX_ADVANCE_ROBOT = config["env"]["max_advance_robot"]
        self.MAX_ROTATION = config["env"]["max_rotation"]
        self.WALL_SEGMENT_SIZE = config["env"]["wall_segment_size"]
        self.SPEED_THRESHOLD = config["env"]["speed_threshold"]

        self.MIN_HUMANS = config["env"]["min_humans"]
        self.MAX_HUMANS = config["env"]["max_humans"]
        assert(self.MIN_HUMANS <= self.MAX_HUMANS), "min_humans should be less than or equal to max_humans"

        self.MIN_TABLES = config["env"]["min_tables"]
        self.MAX_TABLES = config["env"]["max_tables"]
        assert(self.MIN_TABLES <= self.MAX_TABLES), "min_tables should be less than or equal to max_tables"
        
        self.MIN_PLANTS = config["env"]["min_plants"]
        self.MAX_PLANTS = config["env"]["max_plants"]
        assert(self.MIN_PLANTS <= self.MAX_PLANTS), "min_plants should be less than or equal to max_plants"

        self.MIN_LAPTOPS = config["env"]["min_laptops"]
        self.MAX_LAPTOPS = config["env"]["max_laptops"]
        assert(self.MIN_LAPTOPS <= self.MAX_LAPTOPS), "min_laptops should be less than or equal to max_laptops"

        self.MIN_H_H_DYNAMIC_INTERACTIONS = config["env"]["min_h_h_dynamic_interactions"]
        self.MAX_H_H_DYNAMIC_INTERACTIONS = config["env"]["max_h_h_dynamic_interactions"]
        assert(self.MIN_H_H_DYNAMIC_INTERACTIONS <= self.MAX_H_H_DYNAMIC_INTERACTIONS)

        self.MIN_H_H_STATIC_INTERACTIONS = config["env"]["min_h_h_static_interactions"]
        self.MAX_H_H_STATIC_INTERACTIONS = config["env"]["max_h_h_static_interactions"]
        assert(self.MIN_H_H_STATIC_INTERACTIONS <= self.MAX_H_H_STATIC_INTERACTIONS)

        self.MIN_HUMAN_IN_H_H_INTERACTIONS = config["env"]["min_human_in_h_h_interactions"]
        self.MAX_HUMAN_IN_H_H_INTERACTIONS = config["env"]["max_human_in_h_h_interactions"]
        assert(self.MIN_HUMAN_IN_H_H_INTERACTIONS <= self.MAX_HUMAN_IN_H_H_INTERACTIONS), "min_human_in_h_h_interactions should be less than or equal to max_human_in_h_h_interactions"

        self.MIN_H_L_INTERACTIONS = config["env"]["min_h_l_interactions"]
        self.MAX_H_L_INTERACTIONS = config["env"]["max_h_l_interactions"]
        assert(self.MIN_H_L_INTERACTIONS <= self.MAX_H_L_INTERACTIONS), "min_h_l_interactions should be lesser than or equal to max_h_l_interactions"

        self.get_padded_observations = config["env"]["get_padded_observations"]
        assert(self.get_padded_observations == True or self.get_padded_observations == False), "get_padded_observations should be either True or False"

        self.set_shape = config["env"]["set_shape"]
        assert(self.set_shape == "random" or self.set_shape == "square" or self.set_shape == "rectangle" or self.set_shape == "L" or self.set_shape == "no-walls"), "set shape can be \"random\", \"square\", \"rectangle\", \"L\", or \"no-walls\""

        self.MIN_MAP_X = config["env"]["min_map_x"]
        self.MAX_MAP_X = config["env"]["max_map_x"]
        self.MIN_MAP_Y = config["env"]["min_map_y"]
        self.MAX_MAP_Y = config["env"]["max_map_y"]

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
        self.MAP_X = random.randint(self.MIN_MAP_X, self.MAX_MAP_X)
        
        if self.shape == "square" or self.shape == "L":
            self.MAP_Y = self.MAP_X
        else :
            self.MAP_Y = random.randint(self.MIN_MAP_Y, self.MAX_MAP_Y)
        
        self.RESOLUTION_X = int(1500 * self.MAP_X/(self.MAP_X + self.MAP_Y))
        self.RESOLUTION_Y = int(1500 * self.MAP_Y/(self.MAP_X + self.MAP_Y))
        self.NUMBER_OF_HUMANS = random.randint(self.MIN_HUMANS, self.MAX_HUMANS)  # number of humans in the env
        self.NUMBER_OF_PLANTS = random.randint(self.MIN_PLANTS, self.MAX_PLANTS)  # number of plants in the env
        self.NUMBER_OF_TABLES = random.randint(self.MIN_TABLES, self.MAX_TABLES)  # number of tables in the env
        self.NUMBER_OF_LAPTOPS = random.randint(self.MIN_LAPTOPS, self.MAX_LAPTOPS)  # number of laptops in the env. Laptops will be sampled on tables
        self.NUMER_OF_H_H_DYNAMIC_INTERACTIONS = random.randint(self.MIN_H_H_DYNAMIC_INTERACTIONS, self.MAX_H_H_DYNAMIC_INTERACTIONS) # number of dynamic human-human interactions
        self.NUMER_OF_H_H_STATIC_INTERACTIONS = random.randint(self.MIN_H_H_STATIC_INTERACTIONS, self.MAX_H_H_STATIC_INTERACTIONS) # number of static human-human interactions
        self.humans_in_h_h_dynamic_interactions = []
        self.humans_in_h_h_static_interactions = []
        for _ in range(self.NUMER_OF_H_H_DYNAMIC_INTERACTIONS):
            self.humans_in_h_h_dynamic_interactions.append(random.randint(self.MIN_HUMAN_IN_H_H_INTERACTIONS, self.MAX_HUMAN_IN_H_H_INTERACTIONS))
        for _ in range(self.NUMER_OF_H_H_STATIC_INTERACTIONS):
            self.humans_in_h_h_static_interactions.append(random.randint(self.MIN_HUMAN_IN_H_H_INTERACTIONS, self.MAX_HUMAN_IN_H_H_INTERACTIONS))
        
        self.NUMBER_OF_H_L_INTERACTIONS = random.randint(self.MIN_H_L_INTERACTIONS, self.MAX_H_L_INTERACTIONS) # number of human laptop interactions
        
        # total humans
        self.total_humans = self.NUMBER_OF_HUMANS
        for i in self.humans_in_h_h_dynamic_interactions: self.total_humans += i
        for i in self.humans_in_h_h_static_interactions: self.total_humans += i
        self.total_humans += self.NUMBER_OF_H_L_INTERACTIONS

        # randomly select the shape
        if self.set_shape == "random":
            self.shape = random.choice(["rectangle", "square", "L"])
        else: self.shape = self.set_shape
        
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
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0, -self.HUMAN_DIAMETER/2, -(self.MAX_ADVANCE_HUMAN + self.MAX_ADVANCE_ROBOT), -self.MAX_ROTATION] * ((self.MAX_HUMANS + self.MAX_H_L_INTERACTIONS + (self.MAX_H_H_DYNAMIC_INTERACTIONS*self.MAX_HUMAN_IN_H_H_INTERACTIONS) + (self.MAX_H_H_STATIC_INTERACTIONS*self.MAX_HUMAN_IN_H_H_INTERACTIONS)) if self.get_padded_observations else self.total_humans), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, self.HUMAN_DIAMETER/2, +(self.MAX_ADVANCE_HUMAN + self.MAX_ADVANCE_ROBOT), +self.MAX_ROTATION] * ((self.MAX_HUMANS + self.MAX_H_L_INTERACTIONS + (self.MAX_H_H_DYNAMIC_INTERACTIONS*self.MAX_HUMAN_IN_H_H_INTERACTIONS) + (self.MAX_H_H_STATIC_INTERACTIONS*self.MAX_HUMAN_IN_H_H_INTERACTIONS)) if self.get_padded_observations else self.total_humans), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 7) * ((self.MAX_HUMANS + self.MAX_H_L_INTERACTIONS + (self.MAX_H_H_DYNAMIC_INTERACTIONS*self.MAX_HUMAN_IN_H_H_INTERACTIONS) + (self.MAX_H_H_STATIC_INTERACTIONS*self.MAX_HUMAN_IN_H_H_INTERACTIONS)) if self.get_padded_observations else self.total_humans),)),
                dtype=np.float32
            ),

            "laptops": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0, -self.LAPTOP_RADIUS, -(self.MAX_ADVANCE_ROBOT), -self.MAX_ROTATION] * ((self.MAX_LAPTOPS + self.MAX_H_L_INTERACTIONS) if self.get_padded_observations else (self.NUMBER_OF_LAPTOPS + self.NUMBER_OF_H_L_INTERACTIONS)), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, self.LAPTOP_RADIUS, +(self.MAX_ADVANCE_ROBOT), +self.MAX_ROTATION] * ((self.MAX_LAPTOPS + self.MAX_H_L_INTERACTIONS) if self.get_padded_observations else (self.NUMBER_OF_LAPTOPS + self.NUMBER_OF_H_L_INTERACTIONS)), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 7)*((self.MAX_LAPTOPS + self.MAX_H_L_INTERACTIONS) if self.get_padded_observations else (self.NUMBER_OF_LAPTOPS + self.NUMBER_OF_H_L_INTERACTIONS)),)),
                dtype=np.float32

            ),

            "tables": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0, -self.TABLE_RADIUS, -(self.MAX_ADVANCE_ROBOT), -self.MAX_ROTATION] * (self.MAX_TABLES if self.get_padded_observations else self.NUMBER_OF_TABLES), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, self.TABLE_RADIUS, +(self.MAX_ADVANCE_ROBOT), +self.MAX_ROTATION] * (self.MAX_TABLES if self.get_padded_observations else self.NUMBER_OF_TABLES), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 7)*(self.MAX_TABLES if self.get_padded_observations else self.NUMBER_OF_TABLES),)),
                dtype=np.float32

            ),

            "plants": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0, -self.PLANT_RADIUS, -(self.MAX_ADVANCE_ROBOT), -self.MAX_ROTATION] * (self.MAX_PLANTS if self.get_padded_observations else self.NUMBER_OF_PLANTS), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, self.PLANT_RADIUS, +(self.MAX_ADVANCE_ROBOT), +self.MAX_ROTATION] * (self.MAX_PLANTS if self.get_padded_observations else self.NUMBER_OF_PLANTS), dtype=np.float32),
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
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0, -self.WALL_SEGMENT_SIZE, -(self.MAX_ADVANCE_ROBOT), -self.MAX_ROTATION] * int(total_segments), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, +self.WALL_SEGMENT_SIZE, +(self.MAX_ADVANCE_ROBOT), +self.MAX_ROTATION] * int(total_segments), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 7)*int(total_segments),)),
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

    def observation_with_cos_sin_rather_than_angle(self, object): 
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


    def get_observation(self):
        """
        Used to get the observations in the robot frame

        Returns:
            numpy.ndarray : observation as described in the observation space.
        """

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
            obs = self.observation_with_cos_sin_rather_than_angle(human)
            human_obs = np.concatenate((human_obs, obs), dtype=np.float32)
        
        for i in (self.moving_interactions + self.static_interactions + self.h_l_interactions):
            if i.name == "human-human-interaction":
                for human in i.humans:
                    obs = self.observation_with_cos_sin_rather_than_angle(human)
                    human_obs = np.concatenate((human_obs, obs), dtype=np.float32)
            elif i.name == "human-laptop-interaction":
                obs = self.observation_with_cos_sin_rather_than_angle(i.human)
                human_obs = np.concatenate((human_obs, obs), dtype=np.float32)
       
        if self.get_padded_observations:
            # padding with zeros
            human_obs = np.concatenate((human_obs, np.zeros(self.observation_space["humans"].shape[0] - human_obs.shape[0])), dtype=np.float32)
        
        # inserting in the dictionary
        d["humans"] = human_obs

    
        # getting the observations of laptops
        laptop_obs = np.array([], dtype=np.float32)
        for laptop in self.laptops:
            obs = self.observation_with_cos_sin_rather_than_angle(laptop)
            laptop_obs = np.concatenate((laptop_obs, obs), dtype=np.float32)
        
        for i in self.h_l_interactions:
            obs = self.observation_with_cos_sin_rather_than_angle(i.laptop)
            laptop_obs = np.concatenate((laptop_obs, obs), dtype=np.float32)
       
        if self.get_padded_observations:
            # padding with zeros
            laptop_obs = np.concatenate((laptop_obs, np.zeros(self.observation_space["laptops"].shape[0] -laptop_obs.shape[0])), dtype=np.float32)
        
        # inserting in the dictionary
        d["laptops"] = laptop_obs
    

        # getting the observations of tables
        table_obs = np.array([], dtype=np.float32)
        for table in self.tables:
            obs = self.observation_with_cos_sin_rather_than_angle(table)
            table_obs = np.concatenate((table_obs, obs), dtype=np.float32)

        if self.get_padded_observations:
            # padding with zeros
            table_obs = np.concatenate((table_obs, np.zeros(self.observation_space["tables"].shape[0] -table_obs.shape[0])), dtype=np.float32)
        
        # inserting in the dictionary
        d["tables"] = table_obs


        # getting the observations of plants
        plant_obs = np.array([], dtype=np.float32)
        for plant in self.plants:
            obs = self.observation_with_cos_sin_rather_than_angle(plant)
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

    
    def get_desired_force(self, human:Human):
        e_d = np.array([(human.goal_x - human.x), (human.goal_y - human.y)], dtype=np.float32)
        if np.linalg.norm(e_d) != 0:
            e_d /= np.linalg.norm(e_d)
        f_d = self.MAX_ADVANCE_HUMAN * e_d
        return f_d

    def get_obstacle_force(self, human:Human, obstacle:Object, r0):
        # perpendicular distance
        distance = 0

        if obstacle.name == "plant" or obstacle.name=="robot":
            distance = np.sqrt((obstacle.x - human.x)**2 + (obstacle.y - human.y)**2) - obstacle.radius - human.width/2
            e_o = np.array([human.x - obstacle.x, human.y - obstacle.y])
            if np.linalg.norm(e_o) != 0:
                e_o /= np.linalg.norm(e_o)
        
        elif obstacle.name == "table" or obstacle.name == "laptop":
            px, py = get_nearest_point_from_rectangle(obstacle.x, obstacle.y, obstacle.length, obstacle.width, obstacle.orientation, human.x, human.y)      
            e_o = np.array([human.x - px, human.y - py])
            if np.linalg.norm(e_o) != 0:
                e_o /= np.linalg.norm(e_o)
            distance = np.sqrt((human.x-px)**2 + (human.y-py)**2) - human.width/2
        
        elif obstacle.name == "wall":
            px, py = get_nearest_point_from_rectangle(obstacle.x, obstacle.y, obstacle.length, obstacle.thickness, obstacle.orientation, human.x, human.y)      
            e_o = np.array([human.x - px, human.y - py])
            if np.linalg.norm(e_o) != 0:
                e_o /= np.linalg.norm(e_o)
            distance = np.sqrt((human.x-px)**2 + (human.y-py)**2) - human.width/2

        else : raise NotImplementedError

        f_o = np.exp(-distance/r0) * e_o
        return f_o

    def get_interaction_force(self, human1:Human, human2:Human, gamma, n, n_prime, lambd):
        e_ij = np.array([human2.x - human1.x, human2.y - human1.y])
        if np.linalg.norm(e_ij) != 0:
            e_ij /= np.linalg.norm(e_ij)

        v_ij = np.array([
            (human2.speed * np.cos(human2.orientation)) - (human1.speed * np.cos(human1.orientation)),
            (human2.speed * np.sin(human2.orientation)) - (human1.speed * np.sin(human1.orientation))
        ])

        D_ij = lambd *  v_ij + e_ij
        B = np.linalg.norm(D_ij) * gamma
        if np.linalg.norm(D_ij) != 0:
            t_ij = D_ij/np.linalg.norm(D_ij)
        theta_ij = np.arccos(np.clip(np.dot(e_ij, t_ij), -1, 1))
        n_ij = np.array([-e_ij[1], e_ij[0]])
        d_ij = np.sqrt((human1.x-human2.x)**2 + (human1.y-human2.y)**2)
        f_ij = -np.exp(-d_ij/B) * (np.exp(-((n_prime*B*theta_ij)**2))*t_ij + np.exp(-((n*B*theta_ij)**2))*n_ij)
        return f_ij

    def compute_velocity(self, human:Human, w1=1/np.sqrt(3), w2=1/np.sqrt(3), w3=1/np.sqrt(3)):
        f = np.array([0, 0], dtype=np.float32)
        f_d = np.zeros(2, dtype=np.float32)
        f_d = self.get_desired_force(human)
        f += w1*f_d

        if human.avoids_robot:
            for obj in self.plants + self.walls + self.tables + self.laptops + [self.robot]:
                f += w2 * self.get_obstacle_force(human, obj, 0.05)

        else:
            for obj in self.plants + self.walls + self.tables + self.laptops:
                f += w2 * self.get_obstacle_force(human, obj, 0.05)

        for other_human in self.humans:
            if other_human == human: continue
            else:
                f += w3 * self.get_interaction_force(human, other_human, 0.25, 1, 1, 1)

        for i in (self.moving_interactions + self.static_interactions + self.h_l_interactions):
            if i.name == "human-human-interaction":
                for other_human in i.humans:
                    f += w3 * self.get_interaction_force(human, other_human, 1, 1, 1, 1)

            elif i.name == "human-laptop-interaction":
                f += w3 * self.get_interaction_force(human, i.human, 1, 1, 1, 1)
        
        velocity = (f/human.mass) * self.TIMESTEP
        if np.linalg.norm(velocity) > self.MAX_ADVANCE_HUMAN:
            if np.linalg.norm(velocity) != 0:
                velocity /= np.linalg.norm(velocity)
            velocity *= self.MAX_ADVANCE_HUMAN

        return velocity


    def get_obstacle_corners(self, obs:Object):
        if obs.name == "laptop" or obs.name == "table":
            return get_coordinates_of_rotated_rectangle(obs.x, obs.y, obs.orientation, obs.length, obs.width)
        
        elif obs.name == "wall":
            return get_coordinates_of_rotated_rectangle(obs.x, obs.y, obs.orientation, obs.length, obs.thickness)
        
        elif obs.name == "plant" or obs.name == "robot":
            return get_square_around_circle(obs.x, obs.y, obs.radius)
        
        elif obs.name == "human-laptop-interaction":
            return get_square_around_circle(obs.human.x, obs.human.y, 2*self.HUMAN_DIAMETER)
        
        elif obs.name == "human":
            return get_square_around_circle(obs.x, obs.y, 2*obs.width)

        else: raise NotImplementedError


    def compute_orca_velocities(self):
        sim = rvo2.PyRVOSimulator(self.TIMESTEP, 2*self.HUMAN_DIAMETER, self.NUMBER_OF_HUMANS, 5, 5, self.HUMAN_DIAMETER/2, self.MAX_ADVANCE_HUMAN)
        humanList = []
        interactionList = []
        for human in self.humans:
            h = sim.addAgent((human.x, human.y))
            pref_vel = np.array([human.goal_x-human.x, human.goal_y-human.y], dtype=np.float32)
            if not np.linalg.norm(pref_vel) == 0:
                pref_vel /= np.linalg.norm(pref_vel)
            pref_vel *= self.MAX_ADVANCE_HUMAN
            sim.setAgentPrefVelocity(h, (pref_vel[0], pref_vel[1]))
            humanList.append(h)

        for obj in self.tables + self.laptops + self.plants + self.walls:
            p = self.get_obstacle_corners(obj)
            sim.addObstacle(p)

        for i in (self.static_interactions + self.h_l_interactions):
            if i.name == "human-laptop-interaction":
                # p = self.get_obstacle_corners(i)
                # sim.addObstacle(p)
                h = sim.addAgent((i.human.x, i.human.y))
                sim.setAgentPrefVelocity(h, (0, 0))
                sim.setAgentNeighborDist(h, 1.5*self.HUMAN_DIAMETER)

            elif i.name == "human-human-interaction" and i.type == "stationary":
                h = sim.addAgent((i.x, i.y))
                sim.setAgentPrefVelocity(h, (0, 0))
                sim.setAgentRadius(h, self.INTERACTION_RADIUS+self.HUMAN_DIAMETER)
                sim.setAgentNeighborDist(h, 1.5*self.INTERACTION_GOAL_RADIUS)

        for i in self.moving_interactions:
            h = sim.addAgent((i.x, i.y))
            sim.setAgentRadius(h, self.INTERACTION_RADIUS+self.HUMAN_DIAMETER)
            sim.setAgentNeighborDist(h, 1.5*self.INTERACTION_GOAL_RADIUS)
            pref_vel = np.array([i.goal_x-i.x, i.goal_y-i.y], dtype=np.float32)
            if not np.linalg.norm(pref_vel) == 0:
                pref_vel /= np.linalg.norm(pref_vel)
            pref_vel *= self.MAX_ADVANCE_HUMAN
            sim.setAgentPrefVelocity(h, (pref_vel[0], pref_vel[1]))
            interactionList.append(h)

        sim.processObstacles()
        sim.doStep()
        
        vels = []
        interaction_vels = []
        for h in humanList:
            vels.append(sim.getAgentVelocity(h))
        
        for h in interactionList:
            interaction_vels.append(sim.getAgentVelocity(h))

        return vels, interaction_vels

    
    def step(self, action_pre, update=True, frame="robot"):
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
            if action[0] < 0:               # Advance must be positive
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
            self.robot.update(self.TIMESTEP)

            # update humans
            vels, interaction_vels = self.compute_orca_velocities()
            for index, human in enumerate(self.humans):
                if(human.goal_x==None or human.goal_y==None):
                    raise AssertionError("Human goal not specified")
                # human.update(self.TIMESTEP)
                if human.policy == "orca":
                    velocity = vels[index]
                elif human.policy == "sfm":
                    velocity = self.compute_velocity(human)
                human.speed = np.linalg.norm(velocity)
                if human.speed < self.SPEED_THRESHOLD: human.speed = 0
                human.update_orientation(atan2(velocity[1], velocity[0]))
                human.update(self.TIMESTEP)

            # updating moving humans in interactions
            for index, i in enumerate(self.moving_interactions):
                i.update(self.TIMESTEP, interaction_vels[index])
            
            # update the goals for humans if they have reached goal
            for i in range(len(self.humans)):
                HALF_SIZE_X = self.MAP_X/2. - self.MARGIN
                HALF_SIZE_Y = self.MAP_Y/2. - self.MARGIN
                if self.humans[i].has_reached_goal:
                    o = self.update_goal(self.HUMAN_GOAL_RADIUS, HALF_SIZE_X, HALF_SIZE_Y, i)
                    if o is not None:
                        self.humans[i].set_goal(o.x, o.y)

            # update goals of interactions
            for index, i in enumerate(self.moving_interactions):
                if i.has_reached_goal:
                    HALF_SIZE_X = self.MAP_X/2. - self.MARGIN
                    HALF_SIZE_Y = self.MAP_Y/2. - self.MARGIN
                    o = self.update_goal(self.INTERACTION_GOAL_RADIUS, HALF_SIZE_X, HALF_SIZE_Y, self.NUMBER_OF_HUMANS+1+index)
                    if o is not None:
                        i.set_goal(o.x, o.y)


        # getting observations
        observation = self.get_observation()

        # computing rewards and done 
        reward, info = self.compute_reward_and_ticks(action)
        done = self.robot_is_done

        # updating the previous observation
        self.prev_observation = observation

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

    def one_step_lookahead(self, action_pre):
        # storing a copy of env
        env_copy = copy.deepcopy(self)
        next_state, reward, done, info = env_copy.step(action_pre)
        del env_copy
        return next_state, reward, done, info

    def sample_goal(self, goal_radius, HALF_SIZE_X, HALF_SIZE_Y):
        start_time = time.time()
        while True:
            if self.check_timeout(start_time):
                break
            goal = Plant(
                id=None,
                x = random.uniform(-HALF_SIZE_X, HALF_SIZE_X),
                y = random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y),
                radius=goal_radius
            )
            
            collides = False
            for obj in (self.objects + self.goals): # check if spawned object collides with any of the exisiting objects. It will not be rendered as a plant.
                if obj is None: continue
                if(goal.collides(obj)):
                    collides = True
                    break

            if collides:
                del goal
            else:
                return goal
        return None

    def update_goal(self, goal_radius, HALF_SIZE_X, HALF_SIZE_Y, index):
        self.goals[index] = self.sample_goal(goal_radius, HALF_SIZE_X, HALF_SIZE_Y)
        return self.goals[index]


    def compute_reward_and_ticks(self, action):
        """
        Function to compute the reward and also calculate if the episode has finished
        """
        self.ticks += 1

        # calculate the distance to the goal
        distance_to_goal = np.sqrt((self.robot.goal_x - self.robot.x)**2 + (self.robot.goal_y - self.robot.y)**2)

        # check for object-robot collisions
        collision = False

        for object in self.humans + self.plants + self.walls + self.tables + self.laptops:
            if(self.robot.collides(object)): 
                collision = True
                
       
        # interaction-robot collision
        for i in (self.moving_interactions + self.static_interactions + self.h_l_interactions):
            if i.collides(self.robot):
                collision = True
                break
        
        
        dmin = float('inf')
        for human in self.humans:
            px = human.x - self.robot.x
            py = human.y - self.robot.y

            vx = human.speed*np.cos(human.orientation) - action[0] * np.cos(action[1] + self.robot.orientation)
            vy = human.speed*np.sin(human.orientation) - action[0] * np.sin(action[1] + self.robot.orientation)

            ex = px + vx * self.TIMESTEP
            ey = py + vy * self.TIMESTEP

            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - self.HUMAN_DIAMETER/2 - self.ROBOT_RADIUS

            if closest_dist < dmin:
                dmin = closest_dist

        for interaction in (self.moving_interactions + self.static_interactions + self.h_l_interactions):
            px = interaction.x - self.robot.x
            py = interaction.y - self.robot.y

            speed = 0
            if interaction.name == "human-human-interaction":
                for h in interaction.humans:
                    speed += h.speed
                speed /= len(interaction.humans)


            vx = speed*np.cos(0) - action[0] * np.cos(action[1] + self.robot.orientation)
            vy = speed*np.sin(0) - action[0] * np.sin(action[1] + self.robot.orientation)

            ex = px + vx * self.TIMESTEP
            ey = py + vy * self.TIMESTEP

            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - self.HUMAN_DIAMETER/2 - self.ROBOT_RADIUS

            if closest_dist < dmin:
                dmin = closest_dist

        info = {
            "OUT_OF_MAP": False,
            "REACHED_GOAL": False,
            "COLLISION": False,
            "MAX_STEPS": False,
            "DISCOMFORT_SNGNN": 0.0,
            "DISCOMFORT_CROWDNAV": 0.0,
            'sngnn_reward': 0.0
        }

        # calculate the reward and update is_done
        if self.MAP_X/2 < self.robot.x or self.robot.x < -self.MAP_X/2 or self.MAP_Y/2 < self.robot.y or self.robot.y < -self.MAP_Y/2:
            self.robot_is_done = True
            reward = self.OUTOFMAP_REWARD
            info["OUT_OF_MAP"] = True

        elif distance_to_goal < self.GOAL_THRESHOLD:
            self.robot_is_done = True
            reward = self.REACH_REWARD
            info["REACHED_GOAL"] = True

        elif collision is True:
            self.robot_is_done = True
            reward = self.COLLISION_REWARD
            info["COLLISION"] = True

        elif self.ticks > self.EPISODE_LENGTH:
            self.robot_is_done = True
            reward = self.MAX_STEPS_REWARD
            info["MAX_STEPS"] = True
        else:
            self.robot_is_done = False

            sngnn_reward = 0.
            
            if self.USE_SNGNN:
                with torch.no_grad():
                    sn = SNScenario((self.ticks * self.TIMESTEP))
                    robot_goal = self.get_robot_frame_coordinates(np.array([[self.robot.goal_x, self.robot.goal_y]])).flatten()
                    sn.add_goal(-robot_goal[1], robot_goal[0])
                    if (10.32*float(action[1])) >= 4: rot = 4
                    elif (10.32*float(action[1])) <= -4: rot = -4
                    else: rot = (10.32*float(action[1]))
                    sn.add_command([min(9.4*float(action[0]), 3.5), 0.0, rot])
                    # print(f"Action linear: {float(action[0])}  Action angular: {action[1]}")
                    id = 1
                    prev_human_obs = self.prev_observation["humans"].reshape(-1,13)
                    prev_plant_obs = self.prev_observation["plants"].reshape(-1,13)
                    prev_laptop_obs = self.prev_observation["laptops"].reshape(-1,13)
                    prev_table_obs = self.prev_observation["tables"].reshape(-1,13)
                    
                    ind_human = 0
                    ind_plant = 0
                    ind_laptop = 0
                    ind_table = 0
                    
                    for laptop in self.laptops:
                        obs = self.observation_with_cos_sin_rather_than_angle(laptop)
                        sn.add_object(
                            otherObject(
                                id, 
                                -obs[7], 
                                obs[6], 
                                -(np.pi/2 + np.arctan2(obs[8], obs[9])), 
                                (prev_laptop_obs[ind_laptop][6]-obs[6]) / (self.TIMESTEP/0.2),
                                (prev_laptop_obs[ind_laptop][7]-obs[7]) / (self.TIMESTEP/0.2),
                                (np.arctan2(prev_laptop_obs[ind_laptop][8], prev_laptop_obs[ind_laptop][9]) - np.arctan2(obs[8], obs[9]))/(self.TIMESTEP/0.2),
                                laptop.length, 
                                laptop.width
                            )
                        )
                        id += 1
                        ind_laptop += 1

                    for human in self.humans:
                        human_obs = self.observation_with_cos_sin_rather_than_angle(human)
                        sn.add_human(
                            otherHuman(
                                id, 
                                -human_obs[7], 
                                human_obs[6], 
                                -(np.pi/2 + np.arctan2(human_obs[8], human_obs[9])), 
                                (prev_human_obs[ind_human][6]-human_obs[6])/(self.TIMESTEP/0.2),
                                (prev_human_obs[ind_human][7]-human_obs[7])/(self.TIMESTEP/0.2), 
                                (np.arctan2(prev_human_obs[ind_human][8], prev_human_obs[ind_human][9]) - np.arctan2(human_obs[8], human_obs[9]))/(self.TIMESTEP/0.2)
                            )
                        )
                        # print(f"dx: {prev_human_obs[ind_human][6]-human_obs[6]} \ndy: {(prev_human_obs[ind_human][7]-human_obs[7])} \nda: {(np.arctan2(prev_human_obs[ind_human][8], prev_human_obs[ind_human][9]) - np.arctan2(human_obs[8], human_obs[9]))*180/np.pi}\n")
                        id += 1
                        ind_human += 1
                    
                    for interaction in self.moving_interactions + self.static_interactions + self.h_l_interactions:
                        if interaction.name == "human-human-interaction":
                            ids = []
                            for human in interaction.humans:
                                obs = self.observation_with_cos_sin_rather_than_angle(human)
                                sn.add_human(
                                    otherHuman(
                                        id, 
                                        -obs[7], 
                                        obs[6], 
                                        -(np.pi/2 + np.arctan2(obs[8], obs[9])), 
                                        (prev_human_obs[ind_human][6]-obs[6])/(self.TIMESTEP/0.2),
                                        (prev_human_obs[ind_human][7]-obs[7])/(self.TIMESTEP/0.2), 
                                        (np.arctan2(prev_human_obs[ind_human][8], prev_human_obs[ind_human][9]) - np.arctan2(obs[8], obs[9]))/(self.TIMESTEP/0.2)
                                    )
                                )
                                ids.append(id)
                                id += 1
                                ind_human += 1
                            for i in range(len(ids)):
                                for j in range(i+1, len(ids)):
                                    sn.add_interaction([ids[i], ids[j]])
                                    sn.add_interaction([ids[j], ids[i]])
                        
                        if interaction.name == "human-laptop-interaction":
                            obs = self.observation_with_cos_sin_rather_than_angle(interaction.human)
                            sn.add_human(
                                otherHuman(
                                    id, 
                                    -obs[7], 
                                    obs[6], 
                                    -(np.pi/2 + np.arctan2(obs[8], obs[9])), 
                                    (prev_human_obs[ind_human][6]-obs[6])/(self.TIMESTEP/0.2),
                                    (prev_human_obs[ind_human][7]-obs[7])/(self.TIMESTEP/0.2), 
                                    (np.arctan2(prev_human_obs[ind_human][8], prev_human_obs[ind_human][9]) - np.arctan2(obs[8], obs[9]))/(self.TIMESTEP/0.2)
                                )
                            )
                            id += 1
                            ind_human += 1
                            obs = self.observation_with_cos_sin_rather_than_angle(interaction.laptop)
                            sn.add_object(
                                otherObject(
                                    id, 
                                    -obs[7], 
                                    obs[6], 
                                    -(np.pi/2 + np.arctan2(obs[8], obs[9])), 
                                    (prev_laptop_obs[ind_laptop][6]-obs[6]) / (self.TIMESTEP/0.2),
                                    (prev_laptop_obs[ind_laptop][7]-obs[7]) / (self.TIMESTEP/0.2),
                                    (np.arctan2(prev_laptop_obs[ind_laptop][8], prev_laptop_obs[ind_laptop][9]) - np.arctan2(obs[8], obs[9]))/(self.TIMESTEP/0.2),
                                    interaction.laptop.length, 
                                    interaction.laptop.width
                                )
                            )
                            sn.add_interaction([id-1, id])
                            id += 1
                            ind_laptop += 1
                    
                    for plant in self.plants:
                        obs = self.observation_with_cos_sin_rather_than_angle(plant)
                        sn.add_object(
                           otherObject(
                                id, 
                                -obs[7], 
                                obs[6], 
                                -(np.pi/2 + np.arctan2(obs[8], obs[9])), 
                                (prev_plant_obs[ind_plant][6]-obs[6]) / (self.TIMESTEP/0.2),
                                (prev_plant_obs[ind_plant][7]-obs[7]) / (self.TIMESTEP/0.2),
                                (np.arctan2(prev_plant_obs[ind_plant][8], prev_plant_obs[ind_plant][9]) - np.arctan2(obs[8], obs[9]))/(self.TIMESTEP/0.2),
                                plant.radius*2, 
                                plant.radius*2
                            )
                        )
                        id += 1
                        ind_plant += 1
                    
                    for table in self.tables:
                        obs = self.observation_with_cos_sin_rather_than_angle(table)
                        sn.add_object(
                            otherObject(
                                id, 
                                -obs[7], 
                                obs[6], 
                                -(np.pi/2 + np.arctan2(obs[8], obs[9])), 
                                (prev_table_obs[ind_table][6]-obs[6]) / (self.TIMESTEP/0.2),
                                (prev_table_obs[ind_table][7]-obs[7]) / (self.TIMESTEP/0.2),
                                (np.arctan2(prev_table_obs[ind_table][8], prev_table_obs[ind_table][9]) - np.arctan2(obs[8], obs[9]))/(self.TIMESTEP/0.2),
                                table.length, 
                                table.width
                            )
                        )
                        id += 1
                        ind_table += 1
                    
                    assert(ind_human == prev_human_obs.shape[0]), "Something wrong with human obs"
                    assert(ind_plant == prev_plant_obs.shape[0]), "Something wrong with plant obs"
                    assert(ind_table == prev_table_obs.shape[0]), "Something wrong with table obs"
                    assert(ind_laptop == prev_laptop_obs.shape[0]), "Something wrong with laptop obs"

                    wall_list = []
                    for wall in self.walls:
                        x1 = wall.x - np.cos(wall.orientation)*wall.length/2
                        x2 = wall.x + np.cos(wall.orientation)*wall.length/2
                        y1 = wall.y - np.sin(wall.orientation)*wall.length/2
                        y2 = wall.y + np.sin(wall.orientation)*wall.length/2
                        a1 = self.get_robot_frame_coordinates(np.array([[x1, y1]])).flatten()
                        a2 = self.get_robot_frame_coordinates(np.array([[x2, y2]])).flatten()
                        wall_list.append({'x1': -a1[1], 'x2': -a2[1], 'y1': a1[0], 'y2': a2[0]})

                    sn.add_room(wall_list)
                    self.sn_sequence.insert(0, sn.to_json())
                    ## Uncomment to write in json file
                
                    # import json
                    # with open("sample1.json", "w") as f:
                    #     f.write("[")
                    #     for i, d in enumerate(self.sn_sequence):
                    #         json.dump(d, f, indent=4)
                    #         if i != len(self.sn_sequence)-1:
                    #             f.write(",\n")
                    #     f.write("]")

                    #     f.close()
                    graph = SocNavDataset(self.sn_sequence, "1", "test", verbose=False)
                    ret_gnn = self.sngnn.predictOneGraph(graph)[0]
                    sngnn_value = float(ret_gnn[0].item())
                    if sngnn_value < 0.:
                        sngnn_value = 0.
                    elif sngnn_value > 1.:
                        sngnn_value = 1.
                    sngnn_reward = (sngnn_value - 1.0)*self.USE_SNGNN
                    info["DISCOMFORT_SNGNN"] = sngnn_value
                    if dmin < self.DISCOMFORT_DISTANCE:
                        info["DISCOMFORT_CROWDNAV"] = (dmin - self.DISCOMFORT_DISTANCE) * self.DISCOMFORT_PENALTY_FACTOR * self.TIMESTEP

            # elif dmin < self.DISCOMFORT_DISTANCE:
            #     # only penalize agent for getting too close if it's visible
            #     # adjust the reward based on FPS
            #     reward = (dmin - self.DISCOMFORT_DISTANCE) * self.DISCOMFORT_PENALTY_FACTOR * self.TIMESTEP
            #     info["DISCOMFORT_CROWDNAV"] = (dmin - self.DISCOMFORT_DISTANCE) * self.DISCOMFORT_PENALTY_FACTOR * self.TIMESTEP

            # ALIVE penalty
            reward = sngnn_reward + self.ALIVE_REWARD
            # use distance to goal in reward
            if self.USE_DISTANCE_TO_GOAL:
                distance_reward = 0.0
                if self.prev_distance is not None:
                    reward += -(distance_to_goal-self.prev_distance)*self.DISTANCE_REWARD_SCALER
                self.prev_distance = distance_to_goal
                reward += distance_reward

                info['distance_reward'] = distance_reward

            info['sngnn_reward'] = sngnn_reward
            info['alive_reward'] = self.ALIVE_REWARD

        # print(reward)

        return reward, info

    def check_timeout(self, start_time):
        if time.time()-start_time >= 30:
            return True
        else:
            return False

    def reset(self) :
        """
        Resets the environment
        """
        start_time = time.time()
        if not self.has_configured:
            raise Exception("reset() called before configuring the env. Please call env.configure(PATH_TO_CONFIG) before calling env.reset()")
        self.cumulative_reward = 0

        # randomly initialize the parameters 
        self.randomize_params()
        self.id = 0

        HALF_SIZE_X = self.MAP_X/2. - self.MARGIN
        HALF_SIZE_Y = self.MAP_Y/2. - self.MARGIN
        
        # keeping track of the scenarios for sngnn reward
        self.sn_sequence = []

        # to keep track of the current objects
        self.objects = []
        self.laptops = []
        self.walls = []
        self.humans = []
        self.plants = []
        self.tables = []
        self.goals = [None for i in range(self.NUMBER_OF_HUMANS + 1)] # goals of all the humans (as of now interactions are not counted.) + goal of the robot
        self.moving_interactions = []  # a list to keep track of moving interactions
        self.static_interactions = []
        self.h_l_interactions = []

        if self.shape == "L":
            # keep the direction of this as well
            self.location = np.random.randint(0,4)
            
            if self.location == 0:
                self.L_X = 2*self.MAP_X/3
                self.L_Y = self.MAP_Y/3
                # top right
                l = Laptop(
                    id=None,
                    x=self.MAP_X/2.0- self.L_X/2.0,
                    y=self.MAP_Y/2.0 - self.L_Y/2.0,
                    width=self.L_Y,
                    length=self.L_X,
                    theta=0
                )
                # adding walls
                w_l8 = Wall(id=self.id, x=self.MAP_X/2 -self.L_X/2, y=self.MAP_Y/2 -self.L_Y, theta=np.pi, length=self.L_X, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l7 = Wall(id=self.id, x=self.MAP_X/2-(self.WALL_THICKNESS/2), y=-self.L_Y/2, theta=np.pi/2, length=self.MAP_Y-self.L_Y, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l6 = Wall(id=self.id, x=self.MAP_X/6, y=-self.MAP_Y/2 + (self.WALL_THICKNESS/2), theta=0, length=2*self.MAP_X/3, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l5 = Wall(id=self.id, x=-self.MAP_X/3, y=-self.MAP_Y/2 + (self.WALL_THICKNESS/2), theta=0, length=self.MAP_X/3, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l4 = Wall(id=self.id, x=-self.MAP_X/2 + (self.WALL_THICKNESS/2), y=-self.MAP_Y/6, theta=-np.pi/2, length=2*self.MAP_Y/3, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l3 = Wall(id=self.id, x=-self.MAP_X/2 + (self.WALL_THICKNESS/2), y=self.MAP_Y/3, theta=-np.pi/2, length=self.MAP_Y/3, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l2 = Wall(id=self.id, x=-self.L_X/2, y=self.MAP_Y/2-(self.WALL_THICKNESS/2), theta=np.pi, length=self.MAP_X-self.L_X, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l1 = Wall(id=self.id, x=self.MAP_X/2 -self.L_X, y=self.MAP_Y/2 -self.L_Y/2, theta=np.pi/2, length=self.L_Y, thickness=self.WALL_THICKNESS)
                self.id+=1

            elif self.location == 1:
                self.L_X = self.MAP_X/3
                self.L_Y = 2*self.MAP_Y/3
                # top left
                l = Laptop(
                    id=None,
                    x=-self.MAP_X/2.0 + self.L_X/2.0,
                    y=self.MAP_Y/2.0 - self.L_Y/2.0,
                    width=self.L_Y,
                    length=self.L_X,
                    theta=0
                )
                # adding walls
                w_l8 = Wall(id=self.id, x=-self.MAP_X/2 + self.L_X, y=self.MAP_Y/2 -self.L_Y/2, theta=np.pi/2, length=self.L_Y, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l7 = Wall(id=self.id, x=self.L_X/2, y=self.MAP_Y/2-(self.WALL_THICKNESS/2), theta=np.pi, length=self.MAP_X-self.L_X, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l6 = Wall(id=self.id, x=self.MAP_X/2-(self.WALL_THICKNESS/2), y=self.MAP_Y/6, theta=np.pi/2, length=2*self.MAP_Y/3, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l5 = Wall(id=self.id, x=self.MAP_X/2-(self.WALL_THICKNESS/2), y=-self.MAP_Y/3, theta=np.pi/2, length=self.MAP_Y/3, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l4 = Wall(id=self.id, x=self.MAP_X/6, y=-self.MAP_Y/2 + (self.WALL_THICKNESS/2), theta=0, length=2*self.MAP_X/3, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l3 = Wall(id=self.id, x=-self.MAP_X/3, y=-self.MAP_Y/2 + (self.WALL_THICKNESS/2), theta=0, length=self.MAP_X/3, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l2 = Wall(id=self.id, x=-self.MAP_X/2+(self.WALL_THICKNESS/2), y=-self.L_Y/2, theta=-np.pi/2, length=self.MAP_Y-self.L_Y, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l1 = Wall(id=self.id, x=-self.MAP_X/2 +self.L_X/2, y=self.MAP_Y/2 -self.L_Y, theta=np.pi, length=self.L_X, thickness=self.WALL_THICKNESS)
                self.id+=1
            
            elif self.location == 2:
                self.L_X = self.MAP_X/3
                self.L_Y = 2*self.MAP_Y/3
                # bottom right
                l = Laptop(
                    id=None,
                    x=self.MAP_X/2.0 - self.L_X/2.0,
                    y=-self.MAP_Y/2.0 + self.L_Y/2.0,
                    width=self.L_Y,
                    length=self.L_X,
                    theta=0
                )
                # adding walls
                w_l8 = Wall(id=self.id, x=self.MAP_X/2 - self.L_X, y=-self.MAP_Y/2 + self.L_Y/2, theta=np.pi/2,length=self.L_Y, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l7 = Wall(id=self.id, x=-self.L_X/2, y=-self.MAP_Y/2+(self.WALL_THICKNESS/2), theta=0, length=self.MAP_X-self.L_X, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l6 = Wall(id=self.id, x=-self.MAP_X/2+(self.WALL_THICKNESS/2), y=-self.MAP_Y/6, theta=-np.pi/2, length=2*self.MAP_Y/3, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l5 = Wall(id=self.id, x=-self.MAP_X/2+(self.WALL_THICKNESS/2), y=self.MAP_Y/3, theta=-np.pi/2, length=self.MAP_Y/3, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l4 = Wall(id=self.id, x=-self.MAP_X/6, y=self.MAP_Y/2-(self.WALL_THICKNESS/2), theta=np.pi, length=2*self.MAP_X/3, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l3 = Wall(id=self.id, x=self.MAP_X/3, y=self.MAP_Y/2-(self.WALL_THICKNESS/2), theta=np.pi, length=self.MAP_X/3, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l2 = Wall(id=self.id, x=self.MAP_X/2-(self.WALL_THICKNESS/2), y=self.L_Y/2, theta=np.pi/2, length=self.MAP_Y-self.L_Y, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l1 = Wall(id=self.id, x=self.MAP_X/2 - self.L_X/2, y=-self.MAP_Y/2 +self.L_Y, theta=0, length=self.L_X, thickness=self.WALL_THICKNESS)
                self.id+=1

            elif self.location == 3:
                self.L_X = 2*self.MAP_X/3
                self.L_Y = self.MAP_Y/3
                # bottom left
                l = Laptop(
                    id=None,
                    x=-self.MAP_X/2.0 + self.L_X/2.0,
                    y=-self.MAP_Y/2.0 + self.L_Y/2.0,
                    width=self.L_Y,
                    length=self.L_X,
                    theta=0
                )
                # adding walls
                w_l8 = Wall(id=self.id, x=-self.MAP_X/2 + self.L_X/2, y=-self.MAP_Y/2 + self.L_Y, theta=0, length=self.L_X, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l7 = Wall(id=self.id, x=-self.MAP_X/2+(self.WALL_THICKNESS/2), y=self.L_Y/2, theta=-np.pi/2, length=self.MAP_Y-self.L_Y, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l6 = Wall(id=self.id, x=-self.MAP_X/6, y=self.MAP_Y/2-(self.WALL_THICKNESS/2), theta=np.pi, length=2*self.MAP_X/3, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l5 = Wall(id=self.id, x=self.MAP_X/3, y=self.MAP_Y/2-(self.WALL_THICKNESS/2), theta=np.pi, length=self.MAP_X/3, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l4 = Wall(id=self.id, x=self.MAP_X/2-(self.WALL_THICKNESS/2), y=self.MAP_Y/6, theta=np.pi/2, length=2*self.MAP_Y/3, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l3 = Wall(id=self.id, x=self.MAP_X/2-(self.WALL_THICKNESS/2), y=-self.MAP_Y/3, theta=np.pi/2, length=self.MAP_Y/3, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l2 = Wall(id=self.id, x=self.L_X/2, y=-self.MAP_Y/2+(self.WALL_THICKNESS/2), theta=0, length=self.MAP_X-self.L_X, thickness=self.WALL_THICKNESS)
                self.id+=1
                w_l1 = Wall(id=self.id, x= -self.MAP_X/2 +self.L_X, y= -self.MAP_Y/2 + self.L_Y/2, theta=-np.pi/2, length=self.L_Y, thickness=self.WALL_THICKNESS)
                self.id+=1

            self.objects.append(l)
            self.walls.append(w_l1)
            self.walls.append(w_l2)
            self.walls.append(w_l3)
            self.walls.append(w_l4)
            self.walls.append(w_l5)
            self.walls.append(w_l6)
            self.walls.append(w_l7)
            self.walls.append(w_l8)
            self.objects.append(w_l1)
            self.objects.append(w_l2)
            self.objects.append(w_l3)
            self.objects.append(w_l4)
            self.objects.append(w_l5)
            self.objects.append(w_l6)
            self.objects.append(w_l7)
            self.objects.append(w_l8)

        # walls (hardcoded to be at the boundaries of the environment)
        elif self.shape != "no-walls":
            w1 = Wall(self.id, self.MAP_X/2-self.WALL_THICKNESS/2, 0, -np.pi/2, self.MAP_Y, self.WALL_THICKNESS)
            self.id+=1
            w2 = Wall(self.id, 0, -self.MAP_Y/2+self.WALL_THICKNESS/2, -np.pi, self.MAP_X, self.WALL_THICKNESS)
            self.id+=1
            w3 = Wall(self.id, -self.MAP_X/2+self.WALL_THICKNESS/2, 0, np.pi/2, self.MAP_Y, self.WALL_THICKNESS)
            self.id+=1
            w4 = Wall(self.id, 0, self.MAP_Y/2-self.WALL_THICKNESS/2, 0, self.MAP_X, self.WALL_THICKNESS)
            self.id+=1
            self.walls.append(w1)
            self.walls.append(w2)
            self.walls.append(w3)
            self.walls.append(w4)
            self.objects.append(w1)
            self.objects.append(w2)
            self.objects.append(w3)
            self.objects.append(w4)


        success = 1
        # robot
        while True:
            if self.check_timeout(start_time):
                print("timed out, starting again")
                success = 0
                break
            
            robot = Robot(
                id=self.id,
                x = random.uniform(-HALF_SIZE_X, HALF_SIZE_X),
                y = random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y),
                theta = random.uniform(-np.pi, np.pi),
                radius = self.ROBOT_RADIUS,
                goal_x = None,
                goal_y = None
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
                self.id += 1
                break
        if not success:
            self.reset()

        # humans
        for i in range(self.NUMBER_OF_HUMANS): # spawn specified number of humans
            while True: # comes out of loop only when spawned object collides with none of current objects
                if self.check_timeout(start_time):
                    print("timed out, starting again")
                    success = 0
                    break
                x = random.uniform(-HALF_SIZE_X, HALF_SIZE_X)
                y = random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y)

                policy = self.HUMAN_POLICY
                if policy == "random": policy = random.choice(["sfm", "orca"])

                human = Human(
                    id=self.id,
                    x=x,
                    y=y,
                    theta=random.uniform(-np.pi, np.pi) ,
                    width=self.HUMAN_DIAMETER,
                    speed=random.uniform(0.0, self.MAX_ADVANCE_HUMAN),
                    goal_radius=self.HUMAN_GOAL_RADIUS,
                    goal_x=None,
                    goal_y=None,
                    policy=policy
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
                    self.id += 1
                    break
            if not success:
                break
        
        if not success:
            self.reset()
        
        # plants
        for i in range(self.NUMBER_OF_PLANTS): # spawn specified number of plants
            
            while True: # comes out of loop only when spawned object collides with none of current objects
                if self.check_timeout(start_time):
                    print("timed out, starting again")
                    success = 0
                    break

                x = random.uniform(-HALF_SIZE_X, HALF_SIZE_X)
                y = random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y)
                        
                plant = Plant(
                    id=self.id,
                    x=x,
                    y=y,
                    radius=self.PLANT_RADIUS
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
                    self.id+=1
                    break
            
            if not success:
                break
        if not success:
            self.reset()

        # tables
        for i in range(self.NUMBER_OF_TABLES): # spawn specified number of tables
            while True: # comes out of loop only when spawned object collides with none of current objects
                if self.check_timeout(start_time):
                    print("timed out, starting again")
                    success = 0
                    break
                x = random.uniform(-HALF_SIZE_X, HALF_SIZE_X)
                y = random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y)
                        
                table = Table(
                    id=self.id,
                    x=x,
                    y=y,
                    theta=random.uniform(-np.pi, np.pi),
                    width=self.TABLE_WIDTH,
                    length=self.TABLE_LENGTH
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
                    self.id += 1
                    break
            if not success:
                break
            
        if not success:
            self.reset()

        # laptops
        if(len(self.tables) == 0):
            "print: No tables found, placing laptops on the floor!"
            for i in range(self.NUMBER_OF_LAPTOPS): # spawn specified number of laptops
                while True: # comes out of loop only when spawned object collides with none of current objects
                    if self.check_timeout(start_time):
                        print("timed out, starting again")
                        success = 0
                        break

                    x = random.uniform(-HALF_SIZE_X, HALF_SIZE_X)
                    y = random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y)
                            
                    laptop = Laptop(
                        id=self.id,
                        x=x,
                        y=y,
                        theta=random.uniform(-np.pi, np.pi),
                        width=self.LAPTOP_WIDTH,
                        length=self.LAPTOP_LENGTH
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
                        self.id += 1
                        break
                if not success:
                    break
            if not success:
                self.reset()
        
        else:
            for _ in range(self.NUMBER_OF_LAPTOPS): # placing laptops on tables
                while True: # comes out of loop only when spawned object collides with none of current objects
                    if self.check_timeout(start_time):
                        print("timed out, starting again")
                        success = 0
                        break
                    i = random.randint(0, len(self.tables)-1)
                    table = self.tables[i]
                    
                    edge = np.random.randint(0, 4)
                    if edge == 0:
                        center = (
                            table.x + np.cos(table.orientation + np.pi/2) * (self.TABLE_WIDTH-self.LAPTOP_WIDTH)/2, 
                            table.y + np.sin(table.orientation + np.pi/2) * (self.TABLE_WIDTH-self.LAPTOP_WIDTH)/2
                        )
                        theta = table.orientation + np.pi
                    
                    elif edge == 1:
                        center = (
                            table.x + np.cos(table.orientation + np.pi) * (self.TABLE_LENGTH-self.LAPTOP_LENGTH)/2, 
                            table.y + np.sin(table.orientation + np.pi) * (self.TABLE_LENGTH-self.LAPTOP_LENGTH)/2
                        )
                        theta = table.orientation - np.pi/2
                    
                    elif edge == 2:
                        center = (
                            table.x + np.cos(table.orientation - np.pi/2) * (self.TABLE_WIDTH-self.LAPTOP_WIDTH)/2, 
                            table.y + np.sin(table.orientation - np.pi/2) * (self.TABLE_WIDTH-self.LAPTOP_WIDTH)/2
                        )
                        theta = table.orientation
                    
                    elif edge == 3:
                        center = (
                            table.x + np.cos(table.orientation) * (self.TABLE_LENGTH-self.LAPTOP_LENGTH)/2, 
                            table.y + np.sin(table.orientation) * (self.TABLE_LENGTH-self.LAPTOP_LENGTH)/2
                        )
                        theta = table.orientation + np.pi/2
                    
                    laptop = Laptop(
                        id=self.id,
                        x=center[0],
                        y=center[1],
                        theta=theta,
                        width=self.LAPTOP_WIDTH,
                        length=self.LAPTOP_LENGTH
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
                        self.id += 1
                        break
                if not success:
                    break
            if not success:
                self.reset()

        # interactions        
        for ind in range(self.NUMER_OF_H_H_DYNAMIC_INTERACTIONS):
            while True: # comes out of loop only when spawned object collides with none of current objects
                if self.check_timeout(start_time):
                    print("timed out, starting again")
                    success = 0
                    break
                x = random.uniform(-HALF_SIZE_X, HALF_SIZE_X)
                y = random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y)
                i = Human_Human_Interaction(
                    x, y, "moving", self.humans_in_h_h_dynamic_interactions[ind], self.INTERACTION_RADIUS, self.HUMAN_DIAMETER, self.MAX_ADVANCE_HUMAN, self.INTERACTION_GOAL_RADIUS, self.INTERACTION_NOISE_VARIANCE
                )

                collides = False
                for obj in self.objects: # it should not collide with any laptop on the table
                    if(i.collides(obj)):
                        collides = True
                        break

                if collides:
                    del i
                else:
                    self.moving_interactions.append(i)
                    self.objects.append(i)
                    for human in i.humans:
                        human.id = self.id
                        self.id += 1
                    break
            if not success:
                break
        if not success:
            self.reset()

        for ind in range(self.NUMER_OF_H_H_STATIC_INTERACTIONS):
            while True: # comes out of loop only when spawned object collides with none of current objects
                if self.check_timeout(start_time):
                    print("timed out, starting again")
                    success = 0
                    break
                x = random.uniform(-HALF_SIZE_X, HALF_SIZE_X)
                y = random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y)
                i = Human_Human_Interaction(
                    x, y, "stationary", self.humans_in_h_h_static_interactions[ind], self.INTERACTION_RADIUS, self.HUMAN_DIAMETER, self.MAX_ADVANCE_HUMAN, self.INTERACTION_GOAL_RADIUS, self.INTERACTION_NOISE_VARIANCE
                )

                collides = False
                for obj in self.objects: # it should not collide with any laptop on the table
                    if(i.collides(obj)):
                        collides = True
                        break

                if collides:
                    del i
                else:
                    self.static_interactions.append(i)
                    self.objects.append(i)
                    for human in i.humans:
                        human.id = self.id
                        self.id += 1
                    break
            if not success:
                break
        if not success:
            self.reset()
        
        for _ in range(self.NUMBER_OF_H_L_INTERACTIONS):
            # sampling a laptop
            while True:
                if self.check_timeout(start_time):
                    print("timed out, starting again")
                    success = 0
                    break
                i = random.randint(0, len(self.tables)-1)
                table = self.tables[i]
                
                edge = np.random.randint(0, 4)
                if edge == 0:
                    center = (
                        table.x + np.cos(table.orientation + np.pi/2) * (self.TABLE_WIDTH-self.LAPTOP_WIDTH)/2, 
                        table.y + np.sin(table.orientation + np.pi/2) * (self.TABLE_WIDTH-self.LAPTOP_WIDTH)/2
                    )
                    theta = table.orientation + np.pi
                
                elif edge == 1:
                    center = (
                        table.x + np.cos(table.orientation + np.pi) * (self.TABLE_LENGTH-self.LAPTOP_LENGTH)/2, 
                        table.y + np.sin(table.orientation + np.pi) * (self.TABLE_LENGTH-self.LAPTOP_LENGTH)/2
                    )
                    theta = table.orientation - np.pi/2
                
                elif edge == 2:
                    center = (
                        table.x + np.cos(table.orientation - np.pi/2) * (self.TABLE_WIDTH-self.LAPTOP_WIDTH)/2, 
                        table.y + np.sin(table.orientation - np.pi/2) * (self.TABLE_WIDTH-self.LAPTOP_WIDTH)/2
                    )
                    theta = table.orientation
                
                elif edge == 3:
                    center = (
                        table.x + np.cos(table.orientation) * (self.TABLE_LENGTH-self.LAPTOP_LENGTH)/2, 
                        table.y + np.sin(table.orientation) * (self.TABLE_LENGTH-self.LAPTOP_LENGTH)/2
                    )
                    theta = table.orientation + np.pi/2

                laptop = Laptop(
                    id=self.id,
                    x=center[0],
                    y=center[1],
                    theta=theta,
                    width=self.LAPTOP_WIDTH,
                    length=self.LAPTOP_LENGTH
                )

                collides = False
                for obj in self.laptops: # it should not collide with any laptop on the table
                    if(laptop.collides(obj)):
                        collides = True
                        break
                
                for interaction in (self.moving_interactions + self.static_interactions + self.h_l_interactions):
                    if interaction.name == "human-laptop-interaction":
                        if(interaction.collides(laptop)):
                            collides = True
                            break

                if collides:
                    del laptop
                
                else:
                    i = Human_Laptop_Interaction(laptop, self.LAPTOP_WIDTH+self.HUMAN_LAPTOP_DISTANCE, self.HUMAN_DIAMETER)
                    c = False
                    for o in self.objects:
                        if i.collides(o, human_only=True):
                            c = True
                            break
                    if c:
                        del i
                    else:
                        self.h_l_interactions.append(i)
                        self.objects.append(i)
                        self.id+=1
                        i.human.id = self.id
                        self.id += 1
                        break
            if not success:
                break
        if not success:
            self.reset()

        # adding goals
        for i in range(len(self.humans)):   
            o = self.sample_goal(self.HUMAN_GOAL_RADIUS, HALF_SIZE_X, HALF_SIZE_Y)
            if o is None:
                print("timed out, starting again")
                success = 0
                break
            self.goals[i] = o
            self.humans[i].set_goal(o.x, o.y)
        if not success:
            self.reset()

        robot_goal = self.sample_goal(self.GOAL_RADIUS, HALF_SIZE_X, HALF_SIZE_Y)
        if robot_goal is None:
            self.reset()
        self.goals[len(self.humans)] = robot_goal
        self.robot.goal_x = robot_goal.x
        self.robot.goal_y = robot_goal.y

        for i in self.moving_interactions:
            o = self.sample_goal(self.INTERACTION_GOAL_RADIUS, HALF_SIZE_X, HALF_SIZE_Y)
            if o is None:
                print("timed out, starting again")
                success = 0
                break
            self.goals.append(o)
            i.set_goal(o.x, o.y)
        if not success:
            self.reset()

        self.robot_is_done = False
        self.ticks = 0

        # all entities in the environment
        self.entities = self.humans + self.tables + self.laptops + self.plants + self.walls
        self.entities.append(self.robot)
        self.count = 0

        obs = self.get_observation()
        self.prev_observation = obs
        return obs

    def render(self, mode="human"):
        """
        Visualizing the environment
        """

        if not self.window_initialised:
            cv2.namedWindow("world", cv2.WINDOW_NORMAL) 
            cv2.resizeWindow("world", int(self.RESOLUTION_VIEW), int(self.RESOLUTION_VIEW))
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

        cv2.circle(self.world_image, (w2px(self.robot.goal_x, self.PIXEL_TO_WORLD_X, self.MAP_X), w2py(self.robot.goal_y, self.PIXEL_TO_WORLD_Y, self.MAP_Y)), int(w2px(self.robot.x + self.GOAL_RADIUS, self.PIXEL_TO_WORLD_X, self.MAP_X) - w2px(self.robot.x, self.PIXEL_TO_WORLD_X, self.MAP_X)), (0, 255, 0), 2)
        
        for human in self.humans:
            cv2.circle(self.world_image, (w2px(human.goal_x, self.PIXEL_TO_WORLD_X, self.MAP_X), w2py(human.goal_y, self.PIXEL_TO_WORLD_Y, self.MAP_Y)), int(w2px(human.x + self.HUMAN_GOAL_RADIUS, self.PIXEL_TO_WORLD_X, self.MAP_X) - w2px(human.x, self.PIXEL_TO_WORLD_X, self.MAP_X)), (120, 0, 0), 2)
        
        for i in self.moving_interactions:
            cv2.circle(self.world_image, (w2px(i.goal_x, self.PIXEL_TO_WORLD_X, self.MAP_X), w2py(i.goal_y, self.PIXEL_TO_WORLD_Y, self.MAP_Y)), int(w2px(i.x + i.goal_radius, self.PIXEL_TO_WORLD_X, self.MAP_X) - w2px(i.x, self.PIXEL_TO_WORLD_X, self.MAP_X)), (0, 0, 255), 2)
        
        for human in self.humans:
            human.draw(self.world_image, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)
        
        self.robot.draw(self.world_image, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)

        for i in (self.moving_interactions + self.static_interactions + self.h_l_interactions):
            i.draw(self.world_image, self.PIXEL_TO_WORLD_X, self.PIXEL_TO_WORLD_Y, self.MAP_X, self.MAP_Y)

        ## uncomment to save the images 
        # cv2.imwrite("img"+str(self.count)+".jpg", self.world_image)
        # self.count+=1

        cv2.imshow("world", self.world_image)
        k = cv2.waitKey(self.MILLISECONDS)
        if k%255 == 27:
            sys.exit(0)

    def close(self):
        pass
