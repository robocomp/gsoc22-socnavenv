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


# env params
MAP_SIZE = 16.0  # size of the map
MARGIN = 0.5  # outer bound
MAX_ADVANCE = 1.4  # maximum linear speed 
MAX_ROTATION = np.pi*2  # maximum angular speed
NUMBER_OF_HUMANS = random.randint(3, 8)  # number of humans in the env
NUMBER_OF_PLANTS = random.randint(2, 4)  # number of plants in the env
NUMBER_OF_TABLES = random.randint(1, 5)  # number of tables in the env
NUMBER_OF_WALLS = 4  # number of walls in the env. Hardcoded as of now to be the four boundaries of the map
NUMBER_OF_LAPTOPS = random.randint(1, 4)  # number of laptops in the env. Laptops will be sampled on tables
# total objects = sum of all the objects
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
WALL_LENGTH = MAP_SIZE  # so that the wall can cover the side of the map.  


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
        
        # to check if the episode has finished
        self.robot_is_done = True
        # for rendering the world to an OpenCV image
        self.world_image = np.zeros((int(RESOLUTION),int(RESOLUTION),3))


        # parameters for integrating multiagent particle environment's forces

        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

        # to initialize the environment
        self.reset()
    
    @property
    def observation_space(self):
        """
        Observation space includes the goal coordinates in the robot's frame and the relative coordinates and speeds (linear & angular) of all the objects in the scenario
        
        Returns:
        numpy.ndarray : the observation space of the environment
        """
        low  = np.array([-MAP_SIZE, -MAP_SIZE] +                                                                # goal:   x, y (in robot frame) (x, y belong to [-MAP_SIZE, +MAP_SIZE] because the robot and the human can be at opposite corners of the map)
                        [-MAP_SIZE, -MAP_SIZE, -1.0, -1.0, -2*MAX_ADVANCE, -MAX_ROTATION]*TOTAL_OBJECTS )   # objects: x, y, sin, cos, speed (linear), ang_vel (in robot frame) (same reason for x, y as above. The speed can be from [-2*MAX_ADVANCE, 2*MAX_ADVANCE].
        
        high = np.array([+MAP_SIZE, +MAP_SIZE] +                                                                # goal:   x, y (in robot frame)
                        [+MAP_SIZE, +MAP_SIZE, +1.0, +1.0, +2*MAX_ADVANCE, +MAX_ROTATION]*TOTAL_OBJECTS )    # objects: x, y, sin, cos, speed (linear), ang_speed(in robot frame)
        
        return spaces.box.Box(low, high, low.shape, np.float32)

    @property
    def action_space(self): # continuous action space 
        """
        Action space contains two parameters viz linear and angular velocity. Both values lie in the range [-1, 1]. Velocities are obtained by converting the linear value to [0, MAX_ADVANCE] and the angular value to [-MAX_ROTATION, +MAX_ROTATION].
        Returns:
        gym.spaces.box.Box : the action space of the environment
        """
        #               adv rot
        low  = np.array([-1, -1])
        high = np.array([+1, +1])
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
        # 
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

            # object's coordinates in the robot frame
            output = np.concatenate(
                        (
                            output,
                            self.get_robot_frame_coordinates(np.array([[object.x, object.y]])).flatten() 
                        )
                    )

            # sin and cos of the relative angle of the object
            output = np.concatenate(
                        (
                            output,
                            np.array([(np.sin(object.orientation - self.robot.orientation)), np.cos(object.orientation - self.robot.orientation)]) 
                        )
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
                        )
                    )
            return output.flatten()

        # goal coordinates in the robot frame
        goal_in_robot_frame = self.get_robot_frame_coordinates(np.array([[self.robot.goal_x, self.robot.goal_y]]))
        goal_obs = goal_in_robot_frame.flatten()
        
        # getting the observations of each object in the environment (other than the robot)
        object_obs = np.array([], dtype=np.float32)
        for object in (self.humans + self.laptops + self.tables + self.walls + self.plants):
            obs = observation_with_cos_sin_rather_than_angle(object)
            object_obs = np.concatenate((object_obs, obs))
        
        return np.concatenate( (goal_obs, object_obs) ).astype(np.float32)
    
    
    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        """
        Calculating environment forces  References : https://github.com/openai/multiagent-particle-envs/blob/master/multiagent/core.py 
        """
       
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        
        # compute actual distance between entities
        delta_pos = np.array([entity_a.x - entity_b.x, entity_a.y - entity_b.y], dtype=np.float32) 
        # minimum allowable distance
        dist = np.sqrt(np.sum(np.square(delta_pos)))

        if entity_a.name == "plant" or entity_a.name == "robot":
            radius_a = entity_a.radius
        elif entity_a.name == "human":
            radius_a = entity_a.width/2
        elif entity_a.name == "wall":
            radius_a = 0
        # approximating the rectangular objects with a circle that circumscribes it
        elif  entity_a.name == "table" or entity_a.name == "laptop":
            radius_a = np.sqrt((entity_a.length/2)**2 + (entity_a.width/2)**2)
        else: raise NotImplementedError

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
                center_x = 2*entity_a.x - entity_b.x
                center_y = entity_b.y
                delta_pos = np.array([center_x - entity_b.x, center_y - entity_b.y], dtype=np.float32) 
            
            elif entity_a.orientation == 0 or entity_a.orientation == np.pi:
                center_x = entity_b.x
                center_y = 2*entity_a.y - entity_b.y 
                delta_pos = np.array([center_x - entity_b.x, center_y - entity_b.y], dtype=np.float32) 

            else : raise NotImplementedError
            radius_a = radius_b
            dist = np.sqrt(np.sum(np.square(delta_pos)))

        elif entity_b.name == "wall":
            if entity_b.orientation == np.pi/2 or entity_b.orientation == -np.pi/2:
                center_x = 2*entity_b.x - entity_a.x
                center_y = entity_a.y
                delta_pos = np.array([entity_a.x - center_x, entity_a.y - center_y], dtype=np.float32) 
            
            elif entity_b.orientation == 0 or entity_b.orientation == np.pi:
                center_x = entity_a.x
                center_y = 2*entity_b.y - entity_a.y 
                delta_pos = np.array([entity_a.x - center_x, entity_a.y - center_y], dtype=np.float32) 

            else : raise NotImplementedError

            radius_b = radius_a
            dist = np.sqrt(np.sum(np.square(delta_pos)))


        dist_min = radius_a + radius_b
        
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if not entity_a.is_static else None  # forces are applied only to dynamic objects
        force_b = -force if not entity_b.is_static else None  # forces are applied only to dynamic objects
        return [force_a, force_b]

    

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force


    def integrate_state(self, p_force):
        # updates the velocities 
        for i,entity in enumerate(self.entities):
            if entity.is_static: continue
            
            if entity.name == "human":
                if (p_force[i] is not None):
                    entity_vel = (p_force[i] / entity.mass) * TIMESTEP
                    entity.update_velocity(entity_vel[0], entity_vel[1], MAX_ADVANCE)
  
    
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

        # calculating environmental forces
        p_force = [None]*len(self.entities)
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)


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

        # check for the goal's distance
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
                    self.robot.update_velocity(entity_vel[0], entity_vel[1], MAX_ADVANCE)
                break

        # calculate the reward and update is_done
        if MAP_SIZE/2 < self.robot.x or self.robot.x < -MAP_SIZE/2 or MAP_SIZE/2 < self.robot.y or self.robot.y < -MAP_SIZE/2:
            self.robot_is_done = True
            reward = OUTOFMAP_REWARD
        elif distance_to_goal < GOAL_THRESHOLD:
            self.robot_is_done = True
            reward = REACH_REWARD
        elif collision is True:
            self.robot_is_done = False
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
        w1 = Wall(0, MAP_SIZE/2, 0, WALL_LENGTH)
        w2 = Wall(MAP_SIZE/2, 0, np.pi/2, WALL_LENGTH)
        w3 = Wall(0, -MAP_SIZE/2, 0, WALL_LENGTH)
        w4 = Wall(-MAP_SIZE/2, 0, np.pi/2, WALL_LENGTH)
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
