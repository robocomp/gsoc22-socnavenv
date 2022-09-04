import gym
from gym import spaces
from socnavenv.envs.socnavenv_v1 import SocNavEnv_v1
from socnavenv.envs.utils.wall import Wall
import numpy as np
import copy

class WorldFrameObservations(gym.Wrapper):
    def __init__(self, env:SocNavEnv_v1) -> None:
        super().__init__(env)
        self.env = env
        self._observation_space = self.ob_space

    def observation_with_cos_sin_rather_than_angle(self, object): 
            """
            Returning the observation for one individual object. Also to get the sin and cos of the angle rather than the angle itself.
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
                            np.array([object.x, object.y]) 
                        ),
                        dtype=np.float32
                    )

            # sin and cos of the relative angle of the object
            output = np.concatenate(
                        (
                            output,
                            np.array([np.sin(object.orientation), np.cos(object.orientation)]) 
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


            # speeds for static objects
            speeds = np.array([0, 0], dtype=np.float32) 
            
            if object.name == "human": # the only dynamic object
                # relative linear speed
                speeds[0] = object.speed*np.cos(object.orientation)   
                speeds[1] = object.speed*np.sin(object.orientation)
            output = np.concatenate(
                        (
                            output,
                            speeds
                        ),
                        dtype=np.float32
                    )
            return output.flatten()


    def get_world_frame_observations(self):

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
            for w in self.env.walls:
                c, l = segment_wall(w, size)    
                for center, length in zip(c, l):
                    obs = np.concatenate((obs, w.one_hot_encoding))
                    obs = np.concatenate((obs, np.array([center[0], center[1]])))
                    obs = np.concatenate((obs, np.array([np.sin(w.orientation), np.cos(w.orientation)])))
                    obs = np.concatenate((obs, np.array([length/2])))
                    speeds = np.array([0, 0], dtype=np.float32)
                    obs = np.concatenate((obs, speeds))
                    obs = obs.flatten().astype(np.float32)
            return obs

        # the observations will go inside this dictionary
        d = {}
        
        # goal coordinates in the robot frame
        goal_in_robot_frame = (np.array([
            self.env.robot.goal_x, 
            self.env.robot.goal_y, 
            self.env.robot.x, 
            self.env.robot.y,
            np.sin(self.env.robot.orientation),
            np.cos(self.env.robot.orientation),
            self.env.robot.linear_vel,
            self.env.robot.angular_vel 
        ], dtype=np.float32))
        # converting into the required shape
        goal_obs = goal_in_robot_frame.flatten()

        # concatenating with the robot's one-hot-encoding
        goal_obs = np.concatenate((self.env.robot.one_hot_encoding, goal_obs), dtype=np.float32)
        # placing it in a dictionary
        d["goal"] = goal_obs

        # getting the observations of humans
        human_obs = np.array([], dtype=np.float32)
        for human in self.env.humans:
            obs = self.observation_with_cos_sin_rather_than_angle(human)
            human_obs = np.concatenate((human_obs, obs), dtype=np.float32)
        
        for i in (self.env.moving_interactions + self.env.static_interactions + self.env.h_l_interactions):
            if i.name == "human-human-interaction":
                for human in i.humans:
                    obs = self.observation_with_cos_sin_rather_than_angle(human)
                    human_obs = np.concatenate((human_obs, obs), dtype=np.float32)
            elif i.name == "human-laptop-interaction":
                obs = self.observation_with_cos_sin_rather_than_angle(i.human)
                human_obs = np.concatenate((human_obs, obs), dtype=np.float32)
       
        if self.env.get_padded_observations:
            # padding with zeros
            human_obs = np.concatenate((human_obs, np.zeros(self.env.observation_space["humans"].shape[0] - human_obs.shape[0])), dtype=np.float32)
        
        # inserting in the dictionary
        d["humans"] = human_obs

    
        # getting the observations of laptops
        laptop_obs = np.array([], dtype=np.float32)
        for laptop in self.env.laptops:
            obs = self.observation_with_cos_sin_rather_than_angle(laptop)
            laptop_obs = np.concatenate((laptop_obs, obs), dtype=np.float32)
        
        for i in self.env.h_l_interactions:
            obs = self.observation_with_cos_sin_rather_than_angle(i.laptop)
            laptop_obs = np.concatenate((laptop_obs, obs), dtype=np.float32)
       
        if self.env.get_padded_observations:
            # padding with zeros
            laptop_obs = np.concatenate((laptop_obs, np.zeros(self.env.observation_space["laptops"].shape[0] -laptop_obs.shape[0])), dtype=np.float32)
        
        # inserting in the dictionary
        d["laptops"] = laptop_obs
    

        # getting the observations of tables
        table_obs = np.array([], dtype=np.float32)
        for table in self.env.tables:
            obs = self.observation_with_cos_sin_rather_than_angle(table)
            table_obs = np.concatenate((table_obs, obs), dtype=np.float32)

        if self.env.get_padded_observations:
            # padding with zeros
            table_obs = np.concatenate((table_obs, np.zeros(self.env.observation_space["tables"].shape[0] -table_obs.shape[0])), dtype=np.float32)
        
        # inserting in the dictionary
        d["tables"] = table_obs


        # getting the observations of plants
        plant_obs = np.array([], dtype=np.float32)
        for plant in self.env.plants:
            obs = self.observation_with_cos_sin_rather_than_angle(plant)
            plant_obs = np.concatenate((plant_obs, obs), dtype=np.float32)

        if self.env.get_padded_observations:
            # padding with zeros
            plant_obs = np.concatenate((plant_obs, np.zeros(self.env.observation_space["plants"].shape[0] -plant_obs.shape[0])), dtype=np.float32)
        
        # inserting in the dictionary
        d["plants"] = plant_obs

        # inserting wall observations to the dictionary
        if not self.env.get_padded_observations:
            d["walls"] = get_wall_observations(self.env.WALL_SEGMENT_SIZE)

        return d

    @property
    def ob_space(self):
        """
        Observation space includes the goal, and the world frame coordinates and speeds (linear & angular) of all the objects (including the robot) in the scenario
        
        Returns:
        gym.spaces.Dict : the observation space of the environment
        """

        d = {

            "goal": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.env.MAP_X/2 , -self.env.MAP_Y/2, -self.env.MAP_X/2, -self.env.MAP_Y/2, -1.0, -1.0, 0.0, -self.env.MAX_ROTATION], dtype=np.float32), 
                high=np.array([1, 1, 1, 1, 1, 1, self.env.MAP_X/2 , self.env.MAP_Y/2, self.env.MAP_X/2, self.env.MAP_Y/2, 1.0, 1.0, self.env.MAX_ADVANCE_ROBOT, self.env.MAX_ROTATION], dtype=np.float32),
                shape=((self.env.robot.one_hot_encoding.shape[0]+8, )),
                dtype=np.float32

            ),

            "humans": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.env.MAP_X/2 , -self.env.MAP_Y/2 , -1.0, -1.0, -self.env.HUMAN_DIAMETER/2, -self.env.MAX_ADVANCE_HUMAN, -self.env.MAX_ADVANCE_HUMAN] * ((self.env.MAX_HUMANS + self.env.MAX_H_L_INTERACTIONS + (self.env.MAX_H_H_DYNAMIC_INTERACTIONS*self.env.MAX_HUMAN_IN_H_H_INTERACTIONS) + (self.env.MAX_H_H_STATIC_INTERACTIONS*self.env.MAX_HUMAN_IN_H_H_INTERACTIONS)) if self.env.get_padded_observations else self.env.total_humans), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.env.MAP_X/2 , +self.env.MAP_Y/2 , 1.0, 1.0, self.env.HUMAN_DIAMETER/2, +self.env.MAX_ADVANCE_HUMAN, +self.env.MAX_ADVANCE_HUMAN] * ((self.env.MAX_HUMANS + self.env.MAX_H_L_INTERACTIONS + (self.env.MAX_H_H_DYNAMIC_INTERACTIONS*self.env.MAX_HUMAN_IN_H_H_INTERACTIONS) + (self.env.MAX_H_H_STATIC_INTERACTIONS*self.env.MAX_HUMAN_IN_H_H_INTERACTIONS)) if self.env.get_padded_observations else self.env.total_humans), dtype=np.float32),
                shape=(((self.env.robot.one_hot_encoding.shape[0] + 7) * ((self.env.MAX_HUMANS + self.env.MAX_H_L_INTERACTIONS + (self.env.MAX_H_H_DYNAMIC_INTERACTIONS*self.env.MAX_HUMAN_IN_H_H_INTERACTIONS) + (self.env.MAX_H_H_STATIC_INTERACTIONS*self.env.MAX_HUMAN_IN_H_H_INTERACTIONS)) if self.env.get_padded_observations else self.env.total_humans),)),
                dtype=np.float32
            ),

            "laptops": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.env.MAP_X/2 , -self.env.MAP_Y/2 , -1.0, -1.0, -self.env.LAPTOP_RADIUS, 0.0, 0.0] * ((self.env.MAX_LAPTOPS + self.env.MAX_H_L_INTERACTIONS) if self.env.get_padded_observations else (self.env.NUMBER_OF_LAPTOPS + self.env.NUMBER_OF_H_L_INTERACTIONS)), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.env.MAP_X/2 , +self.env.MAP_Y/2 , 1.0, 1.0, self.env.LAPTOP_RADIUS, 0.0, 0.0] * ((self.env.MAX_LAPTOPS + self.env.MAX_H_L_INTERACTIONS) if self.env.get_padded_observations else (self.env.NUMBER_OF_LAPTOPS + self.env.NUMBER_OF_H_L_INTERACTIONS)), dtype=np.float32),
                shape=(((self.env.robot.one_hot_encoding.shape[0] + 7)*((self.env.MAX_LAPTOPS + self.env.MAX_H_L_INTERACTIONS) if self.env.get_padded_observations else (self.env.NUMBER_OF_LAPTOPS + self.env.NUMBER_OF_H_L_INTERACTIONS)),)),
                dtype=np.float32

            ),

            "tables": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.env.MAP_X/2 , -self.env.MAP_Y/2 , -1.0, -1.0, -self.env.TABLE_RADIUS, 0.0, 0.0] * (self.env.MAX_TABLES if self.env.get_padded_observations else self.env.NUMBER_OF_TABLES), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.env.MAP_X/2 , +self.env.MAP_Y/2 , 1.0, 1.0, self.env.TABLE_RADIUS, 0.0, 0.0] * (self.env.MAX_TABLES if self.env.get_padded_observations else self.env.NUMBER_OF_TABLES), dtype=np.float32),
                shape=(((self.env.robot.one_hot_encoding.shape[0] + 7)*(self.env.MAX_TABLES if self.env.get_padded_observations else self.env.NUMBER_OF_TABLES),)),
                dtype=np.float32

            ),

            "plants": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.env.MAP_X/2 , -self.env.MAP_Y/2 , -1.0, -1.0, -self.env.PLANT_RADIUS, 0.0, 0.0] * (self.env.MAX_PLANTS if self.env.get_padded_observations else self.env.NUMBER_OF_PLANTS), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.env.MAP_X/2 , +self.env.MAP_Y/2 , 1.0, 1.0, self.env.PLANT_RADIUS, 0.0, 0.0] * (self.env.MAX_PLANTS if self.env.get_padded_observations else self.env.NUMBER_OF_PLANTS), dtype=np.float32),
                shape=(((self.env.robot.one_hot_encoding.shape[0] + 7)*(self.env.MAX_PLANTS if self.env.get_padded_observations else self.env.NUMBER_OF_PLANTS),)),
                dtype=np.float32

            ),
        }

        if not self.env.get_padded_observations:
            total_segments = 0
            for w in self.env.walls:
                total_segments += w.length//self.env.WALL_SEGMENT_SIZE
                if w.length % self.env.WALL_SEGMENT_SIZE != 0: total_segments += 1
            
            d["walls"] = spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.env.MAP_X/2 , -self.env.MAP_Y/2 , -1.0, -1.0, -self.env.WALL_SEGMENT_SIZE, 0.0, 0.0] * int(total_segments), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.env.MAP_X/2 , +self.env.MAP_Y/2 , 1.0, 1.0, +self.env.WALL_SEGMENT_SIZE, 0.0, 0.0] * int(total_segments), dtype=np.float32),
                shape=(((self.env.robot.one_hot_encoding.shape[0] + 7)*int(total_segments),)),
                dtype=np.float32
            )

        return spaces.Dict(d)

    def step(self, action_pre):
        _, reward, done, info = self.env.step(action_pre)
        obs = self.get_world_frame_observations()
        return obs, reward, done, info

    def reset(self):
        self.env.reset()
        obs = self.get_world_frame_observations()
        self._observation_space = self.ob_space
        return obs

    def one_step_lookahead(self, action_pre):
        # storing a copy of env
        env_copy = copy.deepcopy(self.env)
        next_state, reward, done, info = env_copy.step(action_pre)
        current_env = copy.deepcopy(self.env)
        self.env = copy.deepcopy(env_copy)
        next_state = self.get_world_frame_observations()
        self.env = copy.deepcopy(current_env)
        del current_env
        del env_copy
        return next_state, reward, done, info