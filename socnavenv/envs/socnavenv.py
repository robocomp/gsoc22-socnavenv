import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys
from gym import spaces
import cv2

RESOLUTION = 700.
RESOLUTION_VIEW = 1000.
MAP_SIZE = 8.0
PIXEL_TO_WORLD = RESOLUTION / MAP_SIZE
MARGIN = 0.5
MAX_TICKS = 250
TIMESTEP = 0.1
ROBOT_RADIUS = 0.15
GOAL_RADIUS = 0.5
GOAL_THRESHOLD = ROBOT_RADIUS + GOAL_RADIUS
NUMBER_OF_HUMANS = 5
HUMAN_THRESHOLD = 0.4
REACH_REWARD = 1.0
OUTOFMAP_REWARD = -0.5
MAXTICKS_REWARD = -0.5
ALIVE_REWARD = -0.00001
COLLISION_REWARD = -1.0
DISTANCE_REWARD_DIVISOR = 1000
MAX_ADVANCE = 1.4
MAX_ROTATION = np.pi*2
MILLISECONDS = 30

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
#
# Not urgent:
# - Improve how the robot moves (actual differential platform)
#

class SocNavEnv(gym.Env):
    metadata = {}
    def __init__(self) -> None:
        super().__init__()
        self.window_initialised = False

        self.ticks = 0
        self.humans = np.zeros((NUMBER_OF_HUMANS,4))  # x, y, orientation, speed
        self.robot = np.zeros((1,3))   # x, y, orientation
        self.goal = np.zeros((1,2))    # x, y

        self.robot_is_done = True
        self.world_image = np.zeros((int(RESOLUTION),int(RESOLUTION),3))

        self.reset()

    @property
    def observation_space(self):
        low  = np.array([-MAP_SIZE, -MAP_SIZE] +                                                                # goal:   x, y (in robot frame) (x, y belong to [-MAP_SIZE, +MAP_SIZE] because the robot and the human can be at opposite corners of the map)
                        [-MAP_SIZE, -MAP_SIZE, -1.0, -1.0, -2*MAX_ADVANCE, -MAX_ROTATION]*NUMBER_OF_HUMANS  )   # humans: x, y, sin, cos, speed (linear), ang_vel (in robot frame) (same reason for x, y as above. The speed can be from [-2*MAX_ADVANCE, 2*MAX_ADVANCE]. This is because the robot and human have their velocity vectors in opposite directions (making 180° angle).)
        
        high = np.array([+MAP_SIZE, +MAP_SIZE] +                                                                # goal:   x, y (in robot frame)
                        [+MAP_SIZE, +MAP_SIZE, +1.0, +1.0, +2*MAX_ADVANCE, +MAX_ROTATION]*NUMBER_OF_HUMANS )    # humans: x, y, sin, cos, speed (linear), ang_speed(in robot frame)
        
        return spaces.box.Box(low, high, low.shape, np.float32)
    
    @property
    def action_space(self): # continuous action space 
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

        tm = np.zeros((3,3), dtype=np.float32)
        tm[2,2] = 1
        tm[0,2] = self.robot[0,0]
        tm[1,2] = self.robot[0,1]
        tm[0,0] = tm[1,1] = np.cos(self.robot[0,2])
        tm[1,0] = np.sin(self.robot[0,2])
        tm[0,1] = -1*np.sin(self.robot[0,2])

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
        def observation_with_cos_sin_rather_than_angle(i): 
            output = np.empty((i.shape[0], i.shape[1]+2))
            output[:, 0:2] = self.get_robot_frame_coordinates(i[:, 0:2])
            output[:, 2] = np.sin(i[:, 2] - self.robot[0,2]) # sin of the angle relative to the robot
            output[:, 3] = np.cos(i[:, 2] - self.robot[0,2]) # cos of the angle relative to the robot
            output[:, 4] = np.sqrt((i[:, 3]*np.cos(i[:, 2]) - action[0]*np.cos(self.robot[0,2]))**2 + (i[:, 3]*np.sin(i[:, 2]) - action[0]*np.sin(self.robot[0,2]))**2) # relative linear speed
            output[:, 5] = -1*action[1] # relative angular speed
            return output.flatten()

        goal_in_robot_frame = self.get_robot_frame_coordinates(np.array([[self.goal[0,0], self.goal[0,1]]]))
        # robot_obs = observation_with_cos_sin_rather_than_angle(self.robot)
        goal_obs = goal_in_robot_frame.flatten()
        humans_obs = observation_with_cos_sin_rather_than_angle(self.humans)
        return np.concatenate( (goal_obs, humans_obs) ).astype(np.float32)


    def step(self, action_pre):
        def process_action(action_pre):
            action = np.array(action_pre)
            action[0] = (action[0]+1.0)/2.0*MAX_ADVANCE   # [-1, +1] --> [0, MAX_ADVANCE]
            action[1] = (action[1]+0.0)/1.0*MAX_ROTATION  # [-1, +1] --> [-MAX_ROTATION, +MAX_ROTATION]

            if action[0] < 0:               # Advance must be negative
                action[0] *= -1
            if action[0] > MAX_ADVANCE:     # Advance must be less or equal MAX_ADVANCE
                action[0] = MAX_ADVANCE
            if action[1]   < -MAX_ROTATION:   # Rotation must be higher than -MAX_ROTATION
                action[1] =  -MAX_ROTATION
            elif action[1] > +MAX_ROTATION:  # Rotation must be lower than +MAX_ROTATION
                action[1] =  +MAX_ROTATION

            return action

        def update_robot_and_humans(self, action):
            # update robot
            moved = action[0] * TIMESTEP # a. speed x time
            xp = self.robot[0,0]+np.cos(self.robot[0,2])*moved
            yp = self.robot[0,1]+np.sin(self.robot[0,2])*moved
            self.robot[0,0] = xp
            self.robot[0,1] = yp
            self.robot[0,2] += action[1]*TIMESTEP # r. speed x time

            # update humans' positions
            for human in range(self.humans.shape[0]):
                moved = self.humans[human,3] * TIMESTEP # speed x time
                xp = self.humans[human,0]+np.cos(self.humans[human,2])*moved
                yp = self.humans[human,1]+np.sin(self.humans[human,2])*moved
                self.humans[human,0] = xp
                self.humans[human,1] = yp

        if self.robot_is_done:
            raise Exception('step call within a finished episode!')

        action = process_action(action_pre)
        update_robot_and_humans(self, action)

        observation = self.get_observation(action)
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
        self.ticks += 1

        # check for the goal's distance
        distance_to_goal = np.linalg.norm(self.robot[0,0:2]-self.goal[0,0:2], ord=2)

        # check for human-robot collisions
        collision_with_a_human = False
        for human in range(self.humans.shape[0]):
            distance_to_human = np.linalg.norm(self.robot[0,0:2]-self.humans[human,0:2], ord=2)
            if distance_to_human < HUMAN_THRESHOLD:
                collision_with_a_human = True
                break

        # calculate the reward and update is_done
        if MAP_SIZE/2 < self.robot[0,0] or self.robot[0,0] < -MAP_SIZE/2 or MAP_SIZE/2 < self.robot[0,1] or self.robot[0,1] < -MAP_SIZE/2:
            self.robot_is_done = True
            reward = OUTOFMAP_REWARD
        elif distance_to_goal < GOAL_THRESHOLD:
            self.robot_is_done = True
            reward = REACH_REWARD
        elif collision_with_a_human is True:
            self.robot_is_done = True
            reward = COLLISION_REWARD
        elif self.ticks > MAX_TICKS:
            self.robot_is_done = True
            reward = MAXTICKS_REWARD
        else:
            self.robot_is_done = False
            reward = -distance_to_goal/DISTANCE_REWARD_DIVISOR + ALIVE_REWARD

        return reward

    def reset(self):
        self.cumulative_reward = 0

        HALF_SIZE = MAP_SIZE/2. - MARGIN
        # robot
        self.robot[0,0] = random.uniform(-HALF_SIZE, HALF_SIZE) # x
        self.robot[0,1] = random.uniform(-HALF_SIZE, HALF_SIZE) # y
        self.robot[0,2] = random.uniform(-np.pi, np.pi)         # orientation

        # goal
        self.goal[0,0] = random.uniform(-HALF_SIZE, HALF_SIZE)  # x
        self.goal[0,1] = random.uniform(-HALF_SIZE, HALF_SIZE)  # y

        # humans
        for i in range(NUMBER_OF_HUMANS):
            self.humans[i,0] = random.uniform(-HALF_SIZE, HALF_SIZE)  # x
            self.humans[i,1] = random.uniform(-HALF_SIZE, HALF_SIZE)  # y
            self.humans[i,2] = random.uniform(-np.pi, np.pi)          # orientation
            self.humans[i,3] = random.uniform(0.0, MAX_ADVANCE)               # speed

        self.robot_is_done = False
        self.ticks = 0

        return self.get_observation()
    
    
    def render(self, mode="human"):
        def w2px(i):
            return int(PIXEL_TO_WORLD*(i+(MAP_SIZE/2)))

        def w2py(i):
            return int(PIXEL_TO_WORLD*((MAP_SIZE/2)-i))

        def draw_oriented_point(image, input_data, colour, radius=0.15, nose=0.1):
            left =   np.array([input_data[0,0]+np.cos(input_data[0,2]+np.pi/2)*radius, input_data[0,1]+np.sin(input_data[0,2]+np.pi/2)*radius])
            right =  np.array([input_data[0,0]+np.cos(input_data[0,2]-np.pi/2)*radius, input_data[0,1]+np.sin(input_data[0,2]-np.pi/2)*radius])
            front =  np.array([input_data[0,0]+np.cos(input_data[0,2])*nose,           input_data[0,1]+np.sin(input_data[0,2])*nose])
            centre = np.array([input_data[0,0],                                        input_data[0,1]])

            cv2.line(image, (w2px(centre[0]), w2py(centre[1])), (w2px( left[0]), w2py( left[1])), colour, 3)
            cv2.line(image, (w2px(centre[0]), w2py(centre[1])), (w2px(right[0]), w2py(right[1])), colour, 3)
            cv2.line(image, (w2px(centre[0]), w2py(centre[1])), (w2px(front[0]), w2py(front[1])), colour, 5)


        if not self.window_initialised:
            cv2.namedWindow("world", cv2.WINDOW_NORMAL) 
            cv2.resizeWindow("world", int(RESOLUTION_VIEW), int(RESOLUTION_VIEW))
            self.window_initialised = True

        # empty the image
        self.world_image = (np.ones((int(RESOLUTION),int(RESOLUTION),3))*255).astype(np.uint8)
        # draws robot
        draw_oriented_point(self.world_image, self.robot, (255, 0, 0))
        # draws goal
        cv2.circle(self.world_image, (w2px(self.goal[0,0]), w2py(self.goal[0,1])), int(GOAL_RADIUS*100.), (0, 255, 0), 2)
        # draws humans
        for human in range(NUMBER_OF_HUMANS):
            draw_oriented_point(self.world_image, self.humans[human:human+1,:], (0, 0, 255))
        # shows image
        cv2.imshow("world", self.world_image)
        k = cv2.waitKey(MILLISECONDS)
        if k%255 == 27:
            sys.exit(0)

    def close(self):
        pass

class DiscreteSocNavEnv(SocNavEnv):
    def __init__(self, advance_split=5, rotation_split=5):
        super().__init__()
        self.advance_split = advance_split
        self.rotation_split = rotation_split
        self.action_map = self.build_action_map(advance_split, rotation_split)

    def build_action_map(self, advance_split, rotation_split):
        advance_grid, rotation_grid = np.meshgrid(
            np.linspace(-1, 1, advance_split),
            np.linspace(-1, 1, rotation_split)
            )
        return np.hstack((
            advance_grid.reshape(( advance_split*rotation_split, 1)),
            rotation_grid.reshape((advance_split*rotation_split, 1))
            ))

    @property
    def action_space(self):
        return spaces.Discrete(self.advance_split*self.rotation_split)

    def step(self, action_idx):
        # print(action_idx, self.action_map[action_idx])
        return SocNavEnv.step(self, self.action_map[action_idx])
