import gym
from gym import spaces
from socnavenv.envs.socnavenv_v1 import SocNavEnv_v1
from socnavenv.envs.utils.wall import Wall
import numpy as np
import copy

class PartialObservations(gym.Wrapper):
    def __init__(self, env: SocNavEnv_v1, fov_angle) -> None:
        """
        fov_angle is assumed to be in radians. The range of vision will be assumed from [-fov_angle/2, +fov_angle/2]. The robot heading is assumed to be where the X-axis lies.
        """
        super().__init__(env)
        self.env = env
        self.fov_angle = fov_angle
        assert(self.fov_angle <= 2*np.pi and self.fov_angle>=0), "Lidar angle should be between 0 and 2*pi"
        self.num_humans = 0
        self.num_plants = 0
        self.num_tables = 0
        self.num_laptops = 0
        self.num_walls = 0
        self._observation_space = self.ob_space


    def lies_in_range(self, obs):
        assert(obs.shape == (13,)), "Wrong shape"
        if np.arctan2(obs[7], obs[6]) >= -self.fov_angle/2 and np.arctan2(obs[7], obs[6])<= self.fov_angle/2:
            return True
        else:
            return False

    def get_partial_observation(self, obs):
        d = {}
        d["goal"] = obs["goal"]
        for entity_name in ["humans", "plants", "tables", "laptops"]:
            o = obs[entity_name].reshape(-1, 13)
            partial_obs = np.array([], dtype=np.float32)
            for i in range(o.shape[0]):
                if self.lies_in_range(o[i]):
                    partial_obs = np.concatenate(
                        (partial_obs, o[i])
                    ).flatten()
            d[entity_name] = partial_obs
            if entity_name == "humans":
                self.num_humans = partial_obs.shape[0]//13
            elif entity_name == "plants":
                self.num_plants = partial_obs.shape[0]//13
            elif entity_name == "tables":
                self.num_tables = partial_obs.shape[0]//13
            elif entity_name == "laptops":
                self.num_laptops = partial_obs.shape[0]//13
        
        if "walls" in obs.keys():
            o = obs["walls"].reshape(-1, 13)
            partial_obs = np.array([], dtype=np.float32)
            for i in range(o.shape[0]):
                if self.lies_in_range(o[i]):
                    partial_obs = np.concatenate(
                        (partial_obs, o[i])
                    ).flatten()
            d["walls"] = partial_obs
            self.num_walls = partial_obs.shape[0]//13

        self._observation_space = self.ob_space
        return d

    def step(self, action_pre):
        obs, reward, done, info = self.env.step(action_pre)
        obs = self.get_partial_observation(obs)
        self._observation_space = self.ob_space
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.get_partial_observation(obs)
        self._observation_space = self.ob_space
        return obs

    def one_step_lookahead(self, action_pre):
        # storing a copy of env
        env_copy = copy.deepcopy(self.env)
        next_state, reward, done, info = env_copy.step(action_pre)
        next_state = self.get_partial_observation(next_state)
        self._observation_space = self.ob_space
        del env_copy
        return next_state, reward, done, info

    @property
    def ob_space(self):
        """
        Observation space includes the goal coordinates in the robot's frame and the relative coordinates and speeds (linear & angular) of all the objects in the scenario
        
        Returns:
        gym.spaces.Dict : the observation space of the environment
        """

        d = {

            "goal": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.env.MAP_X * np.sqrt(2), -self.env.MAP_Y * np.sqrt(2)], dtype=np.float32), 
                high=np.array([1, 1, 1, 1, 1, 1, +self.env.MAP_X * np.sqrt(2), +self.env.MAP_Y * np.sqrt(2)], dtype=np.float32),
                shape=((self.env.robot.one_hot_encoding.shape[0]+2, )),
                dtype=np.float32

            ),

            "humans": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.env.MAP_X * np.sqrt(2), -self.env.MAP_Y * np.sqrt(2), -1.0, -1.0, -self.env.HUMAN_DIAMETER/2, -(self.env.MAX_ADVANCE_HUMAN + self.env.MAX_ADVANCE_ROBOT), -self.env.MAX_ROTATION] * (self.num_humans), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.env.MAP_X * np.sqrt(2), +self.env.MAP_Y * np.sqrt(2), 1.0, 1.0, self.env.HUMAN_DIAMETER/2, +(self.env.MAX_ADVANCE_HUMAN + self.env.MAX_ADVANCE_ROBOT), +self.env.MAX_ROTATION] * (self.num_humans), dtype=np.float32),
                shape=(((self.env.robot.one_hot_encoding.shape[0] + 7) * (self.num_humans),)),
                dtype=np.float32
            ),

            "laptops": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.env.MAP_X * np.sqrt(2), -self.env.MAP_Y * np.sqrt(2), -1.0, -1.0, -self.env.LAPTOP_RADIUS, -(self.env.MAX_ADVANCE_ROBOT), -self.env.MAX_ROTATION] * (self.num_laptops), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.env.MAP_X * np.sqrt(2), +self.env.MAP_Y * np.sqrt(2), 1.0, 1.0, self.env.LAPTOP_RADIUS, +(self.env.MAX_ADVANCE_ROBOT), +self.env.MAX_ROTATION] * (self.num_laptops), dtype=np.float32),
                shape=(((self.env.robot.one_hot_encoding.shape[0] + 7)*(self.num_laptops),)),
                dtype=np.float32

            ),

            "tables": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.env.MAP_X * np.sqrt(2), -self.env.MAP_Y * np.sqrt(2), -1.0, -1.0, -self.env.TABLE_RADIUS, -(self.env.MAX_ADVANCE_ROBOT), -self.env.MAX_ROTATION] * (self.num_tables), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.env.MAP_X * np.sqrt(2), +self.env.MAP_Y * np.sqrt(2), 1.0, 1.0, self.env.TABLE_RADIUS, +(self.env.MAX_ADVANCE_ROBOT), +self.env.MAX_ROTATION] * (self.num_tables), dtype=np.float32),
                shape=(((self.env.robot.one_hot_encoding.shape[0] + 7)*(self.num_tables),)),
                dtype=np.float32

            ),

            "plants": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.env.MAP_X * np.sqrt(2), -self.env.MAP_Y * np.sqrt(2), -1.0, -1.0, -self.env.PLANT_RADIUS, -(self.env.MAX_ADVANCE_ROBOT), -self.env.MAX_ROTATION] * (self.num_plants), dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.env.MAP_X * np.sqrt(2), +self.env.MAP_Y * np.sqrt(2), 1.0, 1.0, self.env.PLANT_RADIUS, +(self.env.MAX_ADVANCE_ROBOT), +self.env.MAX_ROTATION] * (self.num_plants), dtype=np.float32),
                shape=(((self.env.robot.one_hot_encoding.shape[0] + 7)*(self.num_plants),)),
                dtype=np.float32

            ),
        }

        if not self.env.get_padded_observations:
            total_segments = 0
            for w in self.env.walls:
                total_segments += w.length//self.env.WALL_SEGMENT_SIZE
                if w.length % self.env.WALL_SEGMENT_SIZE != 0: total_segments += 1
            
            d["walls"] = spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.env.MAP_X * np.sqrt(2), -self.env.MAP_Y * np.sqrt(2), -1.0, -1.0, -self.env.WALL_SEGMENT_SIZE, -(self.env.MAX_ADVANCE_ROBOT), -self.env.MAX_ROTATION] * self.num_walls, dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, +self.env.MAP_X * np.sqrt(2), +self.env.MAP_Y * np.sqrt(2), 1.0, 1.0, +self.env.WALL_SEGMENT_SIZE, +(self.env.MAX_ADVANCE_ROBOT), +self.env.MAX_ROTATION] * self.num_walls, dtype=np.float32),
                shape=(((self.env.robot.one_hot_encoding.shape[0] + 7)*self.num_walls,)),
                dtype=np.float32
            )

        return spaces.Dict(d)