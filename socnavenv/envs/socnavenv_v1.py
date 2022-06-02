import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys
from gym import spaces
import cv2
from paddle import uniform

from socnavenv.envs.utils.human import Human
from socnavenv.envs.utils.laptop import Laptop
from socnavenv.envs.utils.plant import Plant
from socnavenv.envs.utils.robot import Robot
from socnavenv.envs.utils.table import Table
from socnavenv.envs.utils.wall import Wall
from socnavenv.envs.utils.object import Object
from socnavenv.envs.utils.utils import w2px, w2py, uniform_circular_sampler


# env params
MAP_SIZE = 8.0
MARGIN = 0.5
NUMBER_OF_HUMANS = 5
NUMBER_OF_PLANTS = 2
NUMBER_OF_TABLES = 1
NUMBER_OF_WALLS = 4
NUMBER_OF_LAPTOPS = 1
TOTAL_OBJECTS = NUMBER_OF_HUMANS + NUMBER_OF_PLANTS + NUMBER_OF_TABLES + NUMBER_OF_LAPTOPS + NUMBER_OF_WALLS

# rendering params
RESOLUTION = 700.
RESOLUTION_VIEW = 1000.
PIXEL_TO_WORLD = RESOLUTION / MAP_SIZE
MILLISECONDS = 30

# episode params
MAX_TICKS = 250
TIMESTEP = 0.1


# rewards
REACH_REWARD = 1.0
OUTOFMAP_REWARD = -0.5
MAXTICKS_REWARD = -0.5
ALIVE_REWARD = -0.00001
COLLISION_REWARD = -1.0
DISTANCE_REWARD_DIVISOR = 1000

# velocity params
MAX_ADVANCE = 1.4
MAX_ROTATION = np.pi*2

# robot params
ROBOT_RADIUS = 0.4
GOAL_RADIUS = 0.5
GOAL_THRESHOLD = ROBOT_RADIUS + GOAL_RADIUS

# human params
HUMAN_THRESHOLD = 0.4
HUMAN_RADIUS = 0.5


# laptop params
LAPTOP_WIDTH=0.35
LAPTOP_LENGTH=0.4

# plant params
PLANT_RADIUS = 0.2

# table params
TABLE_LENGTH = 3
TABLE_WIDTH = 2

# wall params
WALL_LENGTH = 8


assert(REACH_REWARD>0)
assert(OUTOFMAP_REWARD<0)
assert(MAXTICKS_REWARD<0)
assert(ALIVE_REWARD<0)
assert(COLLISION_REWARD<0)
assert(DISTANCE_REWARD_DIVISOR>1)
assert(MAX_TICKS>1)
assert(HUMAN_THRESHOLD>0)
assert(NUMBER_OF_HUMANS>0)
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

        self.ticks = 0
        self.humans = [] # humans in the scene
        self.laptops = [] # laptops in the scene
        self.walls = []  # walls in the scene
        self.plants = []  # plants in the scene
        self.tables = []  # tables in the scene
        self.robot:Robot = None

        self.robot_is_done = True
        self.world_image = np.zeros((int(RESOLUTION),int(RESOLUTION),3))

        self.reset()
    
    @property
    def observation_space(self):
        """
        Returns the observation space of the environment.
        Observations include the goal coordinates in the robot's frame and the relative coordinates and speeds (linear & angular) of all the objects in the scenario
        """
        low  = np.array([-MAP_SIZE, -MAP_SIZE] +                                                                # goal:   x, y (in robot frame) (x, y belong to [-MAP_SIZE, +MAP_SIZE] because the robot and the human can be at opposite corners of the map)
                        [-MAP_SIZE, -MAP_SIZE, -1.0, -1.0, -2*MAX_ADVANCE, -MAX_ROTATION]*TOTAL_OBJECTS )   # objects: x, y, sin, cos, speed (linear), ang_vel (in robot frame) (same reason for x, y as above. The speed can be from [-2*MAX_ADVANCE, 2*MAX_ADVANCE].
        
        high = np.array([+MAP_SIZE, +MAP_SIZE] +                                                                # goal:   x, y (in robot frame)
                        [+MAP_SIZE, +MAP_SIZE, +1.0, +1.0, +2*MAX_ADVANCE, +MAX_ROTATION]*TOTAL_OBJECTS )    # objects: x, y, sin, cos, speed (linear), ang_speed(in robot frame)
        
        return spaces.box.Box(low, high, low.shape, np.float32)

    @property
    def action_space(self): # continuous action space 
        """
        Returns the action space of the environment
        """
        #               adv rot
        low  = np.array([-1, -1])
        high = np.array([+1, +1])
        return spaces.box.Box(low, high, low.shape, np.float32)

    @property
    def done(self):
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
        """
        assert(self.robot.x is not None and self.robot.y is not None and self.robot.orientation is not None), "Robot coordinates or orientation are None type"
        tm = np.zeros((3,3), dtype=np.float32)
        tm[2,2] = 1
        tm[0,2] = self.robot.x
        tm[1,2] = self.robot.y
        tm[0,0] = tm[1,1] = np.cos(self.robot.orientation)
        tm[1,0] = np.sin(self.robot.orientation)
        tm[0,1] = -1*np.sin(self.robot.orientation)

        return np.linalg.inv(tm)

    def get_robot_frame_coordinates(self, coord):
        """
        coord: coordinate input in the world frame expressed as np.array([[x,y]]).
        If multiple coordinates present, give input as 
        np.array([[x1,y1], 
                  [x2,y2],
                  ...
                  [xn,yn]])
        """
        homogeneous_coordinates = np.c_[coord, np.ones((coord.shape[0], 1))]
        coord_in_robot_frame = (self.transformation_matrix@homogeneous_coordinates.T).T
        return coord_in_robot_frame[:, 0:2]

    def get_observation(self, action=np.zeros(2)):
        """
        Returns the observations in the robot frame
        """
        def observation_with_cos_sin_rather_than_angle(object): 
            assert((object.x is not None) and (object.y is not None) and (object.orientation is not None)), f"{object.name}'s coordinates or orientation are None type"
            output = np.array([], dtype=np.float32)
            output = np.concatenate(
                        (
                            output,
                            self.get_robot_frame_coordinates(np.array([[object.x, object.y]])).flatten() # coordinates of the object in robot frame
                        )
                    )

            output = np.concatenate(
                        (
                            output,
                            np.array([(np.sin(object.orientation - self.robot.orientation)), np.cos(object.orientation - self.robot.orientation)]) # sin and cos of the relative angle
                        )
                    )

            relative_speeds = np.array([-action[0], -action[1]], dtype=np.float32) # relative speeds for static objects
            
            if object.name == "human": # the only dynamic object
                relative_speeds[0] = np.sqrt((object.speed*np.cos(object.orientation) - action[0]*np.cos(self.robot.orientation))**2 + (object.speed*np.sin(object.orientation) - action[0]*np.sin(self.robot.orientation))**2) # relative linear speed
            
            output = np.concatenate(
                        (
                            output,
                            relative_speeds
                        )
                    )
            return output.flatten()

        goal_in_robot_frame = self.get_robot_frame_coordinates(np.array([[self.robot.goal_x, self.robot.goal_y]]))
        goal_obs = goal_in_robot_frame.flatten()
        
        object_obs = np.array([], dtype=np.float32)
        for object in (self.humans + self.laptops + self.tables + self.walls + self.plants):
            obs = observation_with_cos_sin_rather_than_angle(object)
            object_obs = np.concatenate((object_obs, obs))
        
        return np.concatenate( (goal_obs, object_obs) ).astype(np.float32)
    
    
    def step(self, action_pre, update=True):
        """
        Input action : numpy array or list
        """
        def process_action(act):
            action = act.astype(np.float32)
            action[0] = (float(action[0]+1.0)/2.0)*MAX_ADVANCE   # [-1, +1] --> [0, MAX_ADVANCE]
            action[1] = (float(action[1]+0.0)/1.0)*MAX_ROTATION  # [-1, +1] --> [-MAX_ROTATION, +MAX_ROTATION]
            if action[0] < 0:               # Advance must be negative
                action[0] *= -1
            if action[0] > MAX_ADVANCE:     # Advance must be less or equal MAX_ADVANCE
                action[0] = MAX_ADVANCE
            if action[1]   < -MAX_ROTATION:   # Rotation must be higher than -MAX_ROTATION
                action[1] =  -MAX_ROTATION
            elif action[1] > +MAX_ROTATION:  # Rotation must be lower than +MAX_ROTATION
                action[1] =  +MAX_ROTATION

            return action

        if(type(action_pre) == list):
            action_pre = np.array(action_pre, dtype=np.float32)

        if self.robot_is_done:
            raise Exception('step call within a finished episode!')
    
        action = process_action(action_pre)

        if update:
            # update robot
            self.robot.update(action[0], action[1], TIMESTEP)

            # update humans
            for human in self.humans:
                human.update(TIMESTEP)

        # getting observations
        observation = self.get_observation(action)

        # computing rewards and done 
        reward = self.compute_reward_and_ticks()
        done = self.robot_is_done
        info = {}

        self.cumulative_reward += reward

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

        # check for the goal's distance
        distance_to_goal = np.sqrt((self.robot.goal_x - self.robot.x)**2 + (self.robot.goal_y - self.robot.y)**2)

        # check for object-robot collisions
        collision = False
        for object in self.humans + self.plants + self.walls + self.tables + self.laptops :
            if(self.robot.collides(object)): 
                collision = True
                break
        
        # stopping humans if they collide with any obstacle other than humans
        for human in self.humans:
            for obj in self.tables + self.walls + self.plants + self.laptops:
                if(human.collides(obj)):
                    human.speed = 0


        # calculate the reward and update is_done
        if MAP_SIZE/2 < self.robot.x or self.robot.x < -MAP_SIZE/2 or MAP_SIZE/2 < self.robot.y or self.robot.y < -MAP_SIZE/2:
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
        HALF_SIZE = MAP_SIZE/2. - MARGIN

        # to keep track of the current objects
        self.objects = []
        
        # robot
        self.robot = Robot(
            x = random.uniform(-HALF_SIZE, HALF_SIZE),
            y = random.uniform(-HALF_SIZE, HALF_SIZE),
            theta = random.uniform(-np.pi, np.pi),
            radius = ROBOT_RADIUS,
            goal_x = random.uniform(-HALF_SIZE, HALF_SIZE),
            goal_y = random.uniform(-HALF_SIZE, HALF_SIZE)
        )
        
        self.objects.append(self.robot)
        self.objects.append(Plant(self.robot.goal_x, self.robot.goal_y, GOAL_RADIUS)) # adding a plant obstacle as the goal so that the other obstacles that are created do not collide with the goal 

        # humans
        self.humans = []
       
        for i in range(NUMBER_OF_HUMANS): # spawn specified number of humans
            while True: # comes out of loop only when spawned object collides with none of current objects
                x = random.uniform(-HALF_SIZE, HALF_SIZE)
                y = random.uniform(-HALF_SIZE, HALF_SIZE)
                        
                human = Human(
                    x=x,
                    y=y,
                    theta=random.uniform(-np.pi, np.pi) ,
                    width=HUMAN_RADIUS,
                    speed=random.uniform(0.0, MAX_ADVANCE)
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
        self.plants = []
        for i in range(NUMBER_OF_PLANTS): # spawn specified number of plants
            while True: # comes out of loop only when spawned object collides with none of current objects
                x = random.uniform(-HALF_SIZE, HALF_SIZE)
                y = random.uniform(-HALF_SIZE, HALF_SIZE)
                        
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

        # walls (hardcoded to be at the boundaries of the environment)
        self.walls = []
        w1 = Wall(0, 4, 0, WALL_LENGTH)
        w2 = Wall(4, 0, np.pi/2, WALL_LENGTH)
        w3 = Wall(0, -4, 0, WALL_LENGTH)
        w4 = Wall(-4, 0, np.pi/2, WALL_LENGTH)
        self.walls.append(w1)
        self.walls.append(w2)
        self.walls.append(w3)
        self.walls.append(w4)
        self.objects.append(w1)
        self.objects.append(w2)
        self.objects.append(w3)
        self.objects.append(w4)

        # tables
        self.tables = []
        for i in range(NUMBER_OF_TABLES): # spawn specified number of tables
            while True: # comes out of loop only when spawned object collides with none of current objects
                x = random.uniform(-HALF_SIZE, HALF_SIZE)
                y = random.uniform(-HALF_SIZE, HALF_SIZE)
                        
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

        self.laptops = []
        if(len(self.tables) == 0):
            "print: No tables found, placing laptops on the floor!"
            for i in range(NUMBER_OF_LAPTOPS): # spawn specified number of laptops
                while True: # comes out of loop only when spawned object collides with none of current objects
                    x = random.uniform(-HALF_SIZE, HALF_SIZE)
                    y = random.uniform(-HALF_SIZE, HALF_SIZE)
                            
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
            for _ in range(NUMBER_OF_LAPTOPS): # placing laptops on tables
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

        return self.get_observation()

    def render(self, mode="human"):
        """
        Visualizing the environment
        """

        if not self.window_initialised:
            cv2.namedWindow("world", cv2.WINDOW_NORMAL) 
            cv2.resizeWindow("world", int(RESOLUTION_VIEW), int(RESOLUTION_VIEW))
            self.window_initialised = True
        
        self.world_image = (np.ones((int(RESOLUTION),int(RESOLUTION),3))*255).astype(np.uint8)


        for table in self.tables:
            table.draw(self.world_image, PIXEL_TO_WORLD, MAP_SIZE)

        for laptop in self.laptops:
            laptop.draw(self.world_image, PIXEL_TO_WORLD, MAP_SIZE)
        
        for wall in self.walls:
            wall.draw(self.world_image, PIXEL_TO_WORLD, MAP_SIZE)
        
        for plant in self.plants:
            plant.draw(self.world_image, PIXEL_TO_WORLD, MAP_SIZE)

        cv2.circle(self.world_image, (w2px(self.robot.goal_x, PIXEL_TO_WORLD, MAP_SIZE), w2py(self.robot.goal_y, PIXEL_TO_WORLD, MAP_SIZE)), int(GOAL_RADIUS*100.), (0, 255, 0), 2)
        
        for human in self.humans:
            human.draw(self.world_image, PIXEL_TO_WORLD, MAP_SIZE)
        
        self.robot.draw(self.world_image, PIXEL_TO_WORLD, MAP_SIZE)

        cv2.imshow("world", self.world_image)
        k = cv2.waitKey(MILLISECONDS)
        if k%255 == 27:
            sys.exit(0)

    def close(self):
        pass
