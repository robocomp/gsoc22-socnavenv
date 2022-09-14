import sys
sys.path.insert(0, ".")
import torch
import torch.nn as nn
import torch.nn.functional as F
import socnavenv
import gym
import numpy as np
import os
import torch.optim as optim
import argparse
import yaml
import copy
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from agents.models import MLP, CrowdNavMemory
from socnavenv.envs.socnavenv_v1 import SocNavEnv_v1
from torch.utils.data import DataLoader
from torch.autograd import Variable
from socnavenv.wrappers import WorldFrameObservations
from socnavenv.envs.utils.utils import get_square_around_circle
import rvo2

class OM_SARL(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state, cell_size, cell_num):
        super(OM_SARL, self).__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = MLP(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = MLP(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = MLP(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = MLP(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = MLP(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)

        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.mlp3(joint_state)
        return value

class CrowdNavAgent:
    def __init__(self, env, config, **kwargs) -> None:
        assert(env is not None and config is not None)
        # initializing the env
        self.env:SocNavEnv_v1 = env
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # agent variables
        self.input_dim = None
        self.self_state_dim = None
        self.mlp1_dims = None
        self.mlp2_dims = None
        self.mlp3_dims = None
        self.attention_dims = None
        self.with_global_state = None
        self.cell_size = None
        self.cell_num = None
        self.om_channel_size = None
        self.buffer_size = None
        self.gamma = None
        self.num_episodes = None
        self.k = None
        self.lr = None
        self.num_batches = None
        self.batch_size = None
        self.target_update_interval = None
        self.render = None
        self.render_freq = None
        self.save_path = None
        self.save_freq = None
        self.epsilon_start = None
        self.epsilon_end = None
        self.epsilon_decay = None
        self.il_episodes = None
        self.run_name = None

        # if variables are set using **kwargs, it would be considered and not the config entry
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise NameError(f"Variable named {k} not defined")
        
        # setting values from config file
        self.configure(self.config)
        self.epsilon = self.epsilon_start

        self.model = OM_SARL(
            self.input_dim,
            self.self_state_dim,
            self.mlp1_dims,
            self.mlp2_dims,
            self.mlp3_dims,
            self.attention_dims,
            self.with_global_state,
            self.cell_size,
            self.cell_num
        ).to(self.device)
        self.model.apply(self.xavier_init_weights)

        self.target_model = OM_SARL(
            self.input_dim,
            self.self_state_dim,
            self.mlp1_dims,
            self.mlp2_dims,
            self.mlp3_dims,
            self.attention_dims,
            self.with_global_state,
            self.cell_size,
            self.cell_num
        ).to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())

        self.experience_replay = CrowdNavMemory(self.buffer_size)

        if self.run_name is not None:
            self.writer = SummaryWriter('runs/'+self.run_name)
        else:
            self.writer = SummaryWriter()

    def xavier_init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    def configure(self, config:str):
        with open(config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)

        if self.input_dim is None:
            self.input_dim = config["input_dim"]
            assert(self.input_dim is not None), "Argument input_dim cannot be None"

        if self.self_state_dim is None:
            self.self_state_dim = config["self_state_dim"]
            assert(self.self_state_dim is not None), "Argument self_state_dim cannot be None"

        if self.mlp1_dims is None:
            self.mlp1_dims = config["mlp1_dims"]
            assert(self.mlp1_dims is not None), "Argument mlp1_dims cannot be None"

        if self.mlp2_dims is None:
            self.mlp2_dims = config["mlp2_dims"]
            assert(self.mlp2_dims is not None), "Argument mlp2_dims cannot be None"

        if self.mlp3_dims is None:
            self.mlp3_dims = config["mlp3_dims"]
            assert(self.mlp3_dims is not None), "Argument mlp3_dims cannot be None"

        if self.attention_dims is None:
            self.attention_dims = config["attention_dims"]
            assert(self.attention_dims is not None), "Argument attention_dims cannot be None"

        if self.with_global_state is None:
            self.with_global_state = config["with_global_state"]
            assert(self.with_global_state is not None), "Argument with_global_state cannot be None"

        if self.cell_size is None:
            self.cell_size = config["cell_size"]
            assert(self.cell_size is not None), "Argument cell_size cannot be None"

        if self.cell_num is None:
            self.cell_num = config["cell_num"]
            assert(self.cell_num is not None), "Argument cell_num cannot be None"

        if self.om_channel_size is None:
            self.om_channel_size = config["om_channel_size"]
            assert(self.om_channel_size is not None), "Argument om_channel_size cannot be None"

        if self.buffer_size is None:
            self.buffer_size = config["buffer_size"]
            assert(self.buffer_size is not None), "Argument buffer_size cannot be None"

        if self.gamma is None:
            self.gamma = config["gamma"]
            assert(self.gamma is not None), "Argument gamma cannot be None"

        if self.num_episodes is None:
            self.num_episodes = config["num_episodes"]
            assert(self.num_episodes is not None), "Argument num_episodes cannot be None"

        if self.k is None:
            self.k = config["k"]
            assert(self.k is not None), "Argument k cannot be None"

        if self.lr is None:
            self.lr = config["lr"]
            assert(self.lr is not None), "Argument lr cannot be None"

        if self.num_batches is None:
            self.num_batches = config["num_batches"]
            assert(self.num_batches is not None), "Argument num_batches cannot be None"

        if self.batch_size is None:
            self.batch_size = config["batch_size"]
            assert(self.batch_size is not None), "Argument batch_size cannot be None"

        if self.target_update_interval is None:
            self.target_update_interval = config["target_update_interval"]
            assert(self.target_update_interval is not None), "Argument target_update_interval cannot be None"

        if self.render is None:
            self.render = config["render"]
            assert(self.render is not None), "Argument render cannot be None"

        if self.render_freq is None:
            self.render_freq = config["render_freq"]
            assert(self.render_freq is not None), "Argument render_freq cannot be None"

        if self.save_path is None:
            self.save_path = config["save_path"]
            assert(self.save_path is not None), "Argument save_path cannot be None"

        if self.save_freq is None:
            self.save_freq = config["save_freq"]
            assert(self.save_freq is not None), "Argument save_freq cannot be None"

        if self.epsilon_start is None:
            self.epsilon_start = config["epsilon_start"]
            assert(self.epsilon_start is not None), "Argument epsilon_start cannot be None"

        if self.epsilon_end is None:
            self.epsilon_end = config["epsilon_end"]
            assert(self.epsilon_end is not None), "Argument epsilon_end cannot be None"

        if self.epsilon_decay is None:
            self.epsilon_decay = config["epsilon_decay"]
            assert(self.epsilon_decay is not None), "Argument epsilon_decay cannot be None"

        if self.il_episodes is None:
            self.il_episodes = config["il_episodes"]
            assert(self.il_episodes is not None), "Argument il_episodes cannot be None"
            
            
    def discrete_to_continuous_action(self, action:int):
        """
        Function to return a continuous space action for a given discrete action
        """
        if action == 0:
            return np.array([0, 0.125], dtype=np.float32) 
        
        elif action == 1:
            return np.array([0, -0.125], dtype=np.float32) 

        elif action == 2:
            return np.array([1, 0.125], dtype=np.float32) 
        
        elif action == 3:
            return np.array([1, -0.125], dtype=np.float32) 

        elif action == 4:
            return np.array([1, 0], dtype=np.float32)

        elif action == 5:
            return np.array([-1, 0], dtype=np.float32)
        
        else:
            raise NotImplementedError

    def continuous_to_discrete_action(self, vel, self_state):
        vx = vel[0]
        vy = vel[1]

        time = self.env.TIMESTEP
        dx = vx * time
        dy = vy * time

        linear_vel = np.sqrt(dx**2 + dy**2)/time
        angular_vel = (np.arctan2(dy, dx) - self_state[8])/time

        # if linear_vel < self.env.MAX_ADVANCE_ROBOT/4:
        #     return 5
        
        # elif linear_vel <= 3*self.env.MAX_ADVANCE_ROBOT/4:
        #     if angular_vel >= 0:
        #         return 0
        #     else:
        #         return 1
        # else:
        #     if abs(angular_vel) < 0.2:
        #         return 4
        #     elif angular_vel > 0:
        #         return 2
        #     else:
        #         return 3
        if linear_vel > self.env.MAX_ADVANCE_ROBOT: linear_vel = self.env.MAX_ADVANCE_ROBOT
        if angular_vel > self.env.MAX_ROTATION: angular_vel = self.env.MAX_ROTATION
        if angular_vel < -self.env.MAX_ROTATION: angular_vel = -self.env.MAX_ROTATION

        linear_vel /= self.env.MAX_ADVANCE_ROBOT
        angular_vel /= self.env.MAX_ROTATION

        return [linear_vel, angular_vel]


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def socnav_to_crowdnav(self, obs):
        """
        Takes in an observation directly from socnavenv.step() and converts it into crowdnav's observation space
        """
        robot_state = (
            obs["goal"][8],
            obs["goal"][9],
            obs["goal"][12]*obs["goal"][11],
            obs["goal"][12]*obs["goal"][10],
            self.env.ROBOT_RADIUS,
            obs["goal"][6],
            obs["goal"][7],
            self.env.MAX_ADVANCE_ROBOT,
            np.arctan2(obs["goal"][10], obs["goal"][11])
        )

        entity_states = []
        e_names = ["humans", "plants", "tables", "laptops"]
        if "walls" in obs.keys(): e_names.append("walls")
        for entity_name in e_names:
            states = obs[entity_name].reshape(-1, 13)
            for i in range(states.shape[0]):
                entity = states[i]
                entity_states.append((
                    entity[0],
                    entity[1],
                    entity[2],
                    entity[3],
                    entity[4],
                    entity[5],
                    entity[6],
                    entity[7],
                    entity[11],
                    entity[12],
                    entity[10]
                ))
        return robot_state, entity_states

    def rotate(self, state):
        """
        Transform the coordinate to agent-centric.
        Input state tensor is of size (batch_size, state_length)

        """
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #  0     1      2     3      4        5     6      7         8       9       10      11      12      13      14      15     16     17     18      19
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))
        
        theta = (state[:, 8] - rot).reshape((batch, -1))
        
        vx1 = (state[:, 17] * torch.cos(rot) + state[:, 18] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 18] * torch.cos(rot) - state[:, 17] * torch.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 15] - state[:, 0]) * torch.cos(rot) + (state[:, 16] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 16] - state[:, 1]) * torch.cos(rot) - (state[:, 15] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 19].reshape((batch, -1))
        radius_sum = radius + radius1
        encoding = state[:, 9:15].reshape(batch, -1)
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 15]).reshape((batch, -1)), (state[:, 1] - state[:, 16]).
                                  reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
        new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, encoding, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
        return new_state

    def build_occupancy_maps(self, entity_states):
        """
        :param entity_states:
        :return: tensor of shape (# enitities, self.cell_num ** 2 * self.om_channels)
        """
        # 'enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5', 'px', 'py', 'vx', 'vy', 'radius'
        #  0        1       2       3        4      5      6     7      8     9     10    
        occupancy_maps = []
        for entity in entity_states:
            other_entities = np.concatenate([np.array([(other_entity[6], other_entity[7], other_entity[8], other_entity[9])])
                                         for other_entity in entity_states if other_entity != entity], axis=0)
            other_px = other_entities[:, 0] - entity[6]
            other_py = other_entities[:, 1] - entity[7]
            # new x-axis is in the direction of entity's velocity
            human_velocity_angle = np.arctan2(entity[9], entity[8])
            other_human_orientation = np.arctan2(other_py, other_px)
            rotation = other_human_orientation - human_velocity_angle
            distance = np.linalg.norm([other_px, other_py], axis=0)
            other_px = np.cos(rotation) * distance
            other_py = np.sin(rotation) * distance

            # compute indices of humans in the grid
            other_x_index = np.floor(other_px / self.cell_size + self.cell_num / 2)
            other_y_index = np.floor(other_py / self.cell_size + self.cell_num / 2)
            other_x_index[other_x_index < 0] = float('-inf')
            other_x_index[other_x_index >= self.cell_num] = float('-inf')
            other_y_index[other_y_index < 0] = float('-inf')
            other_y_index[other_y_index >= self.cell_num] = float('-inf')
            grid_indices = self.cell_num * other_y_index + other_x_index
            occupancy_map = np.isin(range(self.cell_num ** 2), grid_indices)
            if self.om_channel_size == 1:
                occupancy_maps.append([occupancy_map.astype(int)])
            else:
                # calculate relative velocity for other agents
                other_human_velocity_angles = np.arctan2(other_entities[:, 3], other_entities[:, 2])
                rotation = other_human_velocity_angles - human_velocity_angle
                speed = np.linalg.norm(other_entities[:, 2:4], axis=1)
                other_vx = np.cos(rotation) * speed
                other_vy = np.sin(rotation) * speed
                dm = [list() for _ in range(self.cell_num ** 2 * self.om_channel_size)]
                for i, index in np.ndenumerate(grid_indices):
                    if index in range(self.cell_num ** 2):
                        if self.om_channel_size == 2:
                            dm[2 * int(index)].append(other_vx[i])
                            dm[2 * int(index) + 1].append(other_vy[i])
                        elif self.om_channel_size == 3:
                            dm[3 * int(index)].append(1)
                            dm[3 * int(index) + 1].append(other_vx[i])
                            dm[3 * int(index) + 2].append(other_vy[i])
                        else:
                            raise NotImplementedError
                for i, cell in enumerate(dm):
                    dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
                occupancy_maps.append([dm])

        return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()

    def transform(self, self_state, entity_states):
        """
        Take the state passed from agent and transform it to the input of value network
        :param state:
        :return: tensor of shape (# of humans, len(state))
        """
        state_tensor = torch.cat([torch.Tensor([self_state + entity_state]).to(self.device)
                                  for entity_state in entity_states], dim=0)
        occupancy_maps = self.build_occupancy_maps(entity_states)
        state_tensor = torch.cat([self.rotate(state_tensor), occupancy_maps.to(self.device)], dim=1)
        return state_tensor

    def get_action(self, state):
        """
        Main function that uses the SARL network, makes the occupancy map and gets the action.
        Args: state (dict): State that is received from SocNavEnv
        Returns: Discrete action to be taken
        """
        self_state, entity_states = self.socnav_to_crowdnav(state)

        occupancy_maps = None
        probability = np.random.random()
        
        # explore
        if probability < self.epsilon:
            max_action = np.random.randint(0, 6)
        # exploit
        else:
            self.action_values = []
            max_value = float('-inf')
            max_action = None

            for action in range(0, 6):
                action_continuous = self.discrete_to_continuous_action(action)
                next_state, reward, done, info = self.env.one_step_lookahead(action_continuous)
                next_self_state, next_entity_states = self.socnav_to_crowdnav(next_state)
                batch_next_states = torch.cat([torch.Tensor([next_self_state + next_entity_state]).to(self.device)
                                              for next_entity_state in next_entity_states], dim=0)
            
            rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)
            if occupancy_maps is None:
                occupancy_maps = self.build_occupancy_maps(next_entity_states).unsqueeze(0)
            rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps.to(self.device)], dim=2)

            # VALUE UPDATE
            next_state_value = self.model(rotated_batch_input).data.item()
            value = reward + pow(self.gamma, self.env.TIMESTEP * self_state[7]) * next_state_value

            if value > max_value:
                max_value = value
                max_action = action

            if max_action is None:
                raise ValueError('Value network is not well trained. ')

        self.last_state = self.transform(self_state, entity_states)
        return max_action

    def get_imitation_learning_action(self, state):
        """
        This will take an action according to ORCA policy
        """

        def get_entity_type(state):
            if state[0] == 1: return "robot"
            elif state[1] == 1: return "human"
            elif state[2] == 1: return "table"
            elif state[3] == 1: return "laptop"
            elif state[4] == 1: return "plant"
            elif state[5] == 1: return "wall"
            else: raise NotImplementedError

        # 'enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5', 'px', 'py', 'vx', 'vy', 'radius'
        #  0        1       2       3        4      5      6     7      8     9     10    
        self_state, entity_states = self.socnav_to_crowdnav(state)
        self.sim = rvo2.PyRVOSimulator(
            self.env.TIMESTEP, 
            self.env.HUMAN_DIAMETER + self.env.ROBOT_RADIUS*2,
            self.env.NUMBER_OF_HUMANS+1,
            5,
            5,
            self.env.HUMAN_DIAMETER/2,
            self.env.MAX_ADVANCE_HUMAN
        )

        # adding robot to the simulator
        r = self.sim.addAgent((self_state[0], self_state[1]))
        pref_vel = np.array([self_state[5]-self_state[0], self_state[6]-self_state[1]], dtype=np.float32)
        if not np.linalg.norm(pref_vel) == 0:
            pref_vel /= np.linalg.norm(pref_vel)
        pref_vel *= self.env.MAX_ADVANCE_ROBOT
        self.sim.setAgentPrefVelocity(r, (pref_vel[0], pref_vel[1]))


        # adding all humans to the simulator, and setting their v_pref
        for entity in entity_states:
            if get_entity_type(entity) == "human":
                h = self.sim.addAgent((entity[6], entity[7]))
                self.sim.setAgentPrefVelocity(h, (entity[8], entity[9]))

            else:
                p = get_square_around_circle(entity[6], entity[7], entity[10])
                self.sim.addObstacle(p)
        self.sim.processObstacles()
        self.sim.doStep()
        vel = self.sim.getAgentVelocity(r)
        action = self.continuous_to_discrete_action(vel, self_state)
        self.last_state = (self_state, entity_states)
        return action


    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.experience_replay is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.transform(state[0], state[1])
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([pow(self.gamma, max(t - i, 0) * self.env.TIMESTEP * self.env.MAX_ADVANCE_ROBOT) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.env.TIMESTEP * self.env.MAX_ADVANCE_ROBOT)
                    value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()
            value = torch.Tensor([value]).to(self.device)
            self.experience_replay.push((state, value))

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def explore(self, k, imitation_learning=False, episode=None, update_memory=True):
        for i in range(k):
            ob = self.env.reset()
            done = False
            states = []
            actions = []
            rewards = []

            steps = 0
            while not done:
                # sampling action using current policy
                if not imitation_learning:
                    action = self.get_action(ob)
                    act_continuous = self.discrete_to_continuous_action(action)
                else:
                    act_continuous = self.get_imitation_learning_action(ob)
                    action = act_continuous
                # taking a step in the environment 
                ob, reward, done, info = self.env.step(act_continuous)

                # storing states, actions, rewards
                states.append(self.last_state)
                actions.append(action)
                rewards.append(reward)

                # incrementing total steps
                steps += 1

                # rendering if reqd
                if self.render and ((i+1) % self.render_freq == 0):
                    self.env.render()

                # add plotting code
                self.episode_reward += reward

                # storing discomforts
                self.discomfort_sngnn += info["DISCOMFORT_SNGNN"]
                self.discomfort_crowdnav += info["DISCOMFORT_CROWDNAV"]

                 # storing whether the agent reached the goal
                if info["REACHED_GOAL"]:
                    self.has_reached_goal += 1
                
                if info["COLLISION"]:
                    self.has_collided = 1
                    steps = self.env.EPISODE_LENGTH
            
            self.steps += steps

            if update_memory:
                if info["COLLISION"] or info["REACHED_GOAL"]:
                    self.update_memory(states, actions, rewards, imitation_learning)

    def update(self, imitation_learning=False):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        
        for _ in range(self.num_batches):
            if not imitation_learning:
                inputs, values = self.experience_replay.sample_batch(self.batch_size)
            else:
                inputs, values = self.experience_replay.sample_batch(len(self.experience_replay))

            inputs = inputs.to(self.device)
            values = values.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, values)
            self.episode_loss += loss.item()

            if imitation_learning:
                print(f"Loss: {loss.item()}")
            self.optimizer.zero_grad()
            loss.backward()
            
            # gradient clipping
            self.total_grad_norm += torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5).item()
            self.optimizer.step()

    def plot(self, episode):
        self.rewards.append(self.episode_reward)
        self.losses.append(self.episode_loss)
        self.exploration_rates.append(self.epsilon)
        self.grad_norms.append(self.total_grad_norm)
        self.successes.append(self.has_reached_goal)
        self.collisions.append(self.has_collided)
        self.steps_to_reach.append(self.steps)
        self.discomforts_sngnn.append(self.discomfort_sngnn)
        self.discomforts_crowdnav.append(self.discomfort_crowdnav)

        if not os.path.isdir(os.path.join(self.save_path, "plots")):
            os.makedirs(os.path.join(self.save_path, "plots"))

        np.save(os.path.join(self.save_path, "plots", "rewards"), np.array(self.rewards), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "losses"), np.array(self.losses), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "exploration_rates"), np.array(self.exploration_rates), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "grad_norms"), np.array(self.grad_norms), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "successes"), np.array(self.successes), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "collisions"), np.array(self.collisions), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "discomfort_sngnn"), np.array(self.discomforts_sngnn), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "discomfort_crowdnav"), np.array(self.discomforts_crowdnav), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "steps_to_reach"), np.array(self.steps_to_reach), allow_pickle=True, fix_imports=True)

        self.writer.add_scalar("reward / epsiode", self.episode_reward, episode)
        self.writer.add_scalar("loss / episode", self.episode_loss, episode)
        self.writer.add_scalar("exploration rate / episode", self.epsilon, episode)
        self.writer.add_scalar("Average total grad norm / episode", (self.total_grad_norm/self.batch_size), episode)
        self.writer.add_scalar("ending in sucess? / episode", self.has_reached_goal, episode)
        self.writer.add_scalar("has collided? / episode", self.has_collided, episode)
        self.writer.add_scalar("Steps to reach goal / episode", self.steps, episode)
        self.writer.add_scalar("Discomfort SNGNN / episode", self.discomfort_sngnn, episode)
        self.writer.add_scalar("Discomfort CrowdNav / episode", self.discomfort_crowdnav, episode)
        self.writer.flush()  
       
    def train(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.criterion = nn.MSELoss().to(self.device)
        self.rewards = []
        self.losses = []
        self.exploration_rates = []
        self.grad_norms = []
        self.successes = []
        self.collisions = []
        self.steps_to_reach = []
        self.discomforts_sngnn = []
        self.discomforts_crowdnav = []


        print("Performing Imitation Learning")
        self.episode_reward = 0
        self.episode_loss = 0
        self.total_grad_norm = 0
        self.has_reached_goal = 0
        self.has_collided = 0
        self.steps = 0
        self.discomfort_sngnn = 0
        self.discomfort_crowdnav = 0
        self.explore(self.il_episodes, imitation_learning=True)
        self.update(imitation_learning=True)
        print("Finished Imitation Learning")

        episode = 0
        while episode < self.num_episodes:
            self.episode_reward = 0
            self.episode_loss = 0
            self.total_grad_norm = 0
            self.has_reached_goal = 0
            self.has_collided = 0
            self.steps = 0
            self.discomfort_sngnn = 0
            self.discomfort_crowdnav = 0

            # decaying epsilon
            if episode < self.epsilon_decay:
                self.epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) / self.epsilon_decay * episode
            
            self.explore(self.k, episode=episode)
            self.episode_reward /= self.k
            self.has_reached_goal /= self.k
            self.has_collided /= self.k
            self.steps /= self.k
            self.discomfort_sngnn /= self.k
            self.discomfort_crowdnav /= self.k
            
            self.update()
            self.episode_loss /= self.num_batches
            self.total_grad_norm /= self.num_batches
            
            
            episode += 1
            if episode % self.target_update_interval == 0:
                self.update_target_model(self.model)

            # plotting using tensorboard
            print(f"Episode {episode} Avg Reward: {self.episode_reward} Avg Loss: {self.episode_loss}")
            self.plot(episode)


if __name__ == "__main__":
    env:SocNavEnv_v1 = gym.make("SocNavEnv-v1")
    env.configure("./configs/empty.yaml")
    env.set_padded_observations(False)
    env = WorldFrameObservations(env)
    agent = CrowdNavAgent(env, "./configs/crowdnav.yaml", run_name="crowdnav")
    agent.train()