import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
import socnavenv
import gym
import numpy as np
import copy
import random
import torch.optim as optim
import os
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter
from agents.models import ExperienceReplay, Transformer, MLP

class Critic_Transformer(nn.Module):
    def __init__(self, input_emb1:int, input_emb2:int, d_model:int, d_k:int, mlp_hidden_layers:list, v_net_layers:list, device) -> None:
        super().__init__()
        # sizes of the first layer in the critic networks should be same as the output of the hidden layer network + 2 (dimension for action space)
        assert(v_net_layers[0]==mlp_hidden_layers[-1]+2)
        
        self.transformer = Transformer(input_emb1, input_emb2, d_model, d_k, mlp_hidden_layers)
        self.critic_network = MLP(v_net_layers[0], v_net_layers[1:])

        self.device = device

    def forward(self, inp1, inp2, action):
        h = self.transformer.forward(inp1, inp2)
        q_a = self.critic_network.forward(torch.cat([h, action], dim=-1).to(self.device))
        return q_a

class Actor_Transformer(nn.Module):
    def __init__(self, input_emb1:int, input_emb2:int, d_model:int, d_k:int, mlp_hidden_layers:list, a_net_layers:list, mean:float, stddev:float, episode_to_explore_till:int, device) -> None:
        super().__init__()
        # sizes of the first layer in the actor networks should be same as the output of the hidden layer network
        assert(a_net_layers[0]==mlp_hidden_layers[-1])
        
        self.mean = mean
        self.stddev = stddev
        self.decay_rate = stddev/episode_to_explore_till
        self.device = device

        self.transformer = Transformer(input_emb1, input_emb2, d_model, d_k, mlp_hidden_layers)
        self.actor_network = MLP(a_net_layers[0], a_net_layers[1:])
        

    def update_stddev(self):
        self.stddev -= self.decay_rate

    def forward(self, inp1, inp2):
        h = self.transformer.forward(inp1, inp2)
        a = self.actor_network.forward(h)
        # adding gaussian noise for exploration
        noise = torch.empty(a.shape).normal_(mean=self.mean,std=self.stddev).to(self.device)
        self.update_stddev()
        action_std = torch.empty(a.shape).normal_(mean=0.0,std=1.0).to(self.device)
        action_continuous = torch.stack([torch.clip(a[:,:,0]+action_std[:,:,0]+noise[:,:,0], -1.0, 1.0), torch.clip(a[:,:,1]+action_std[:,:,1]+noise[:,:,1], -1, 1)])
        return action_continuous

class DDPG_Transformer_Agent:
    def __init__(self, env:gym.Env, config:str, **kwargs) -> None:
        assert(env is not None and config is not None)
        # initializing the env
        self.env = env
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # agent variables
        self.input_emb1 = None
        self.input_emb2 = None
        self.d_model = None
        self.d_k = None
        self.mlp_hidden_layers = None
        self.v_net_layers = None
        self.a_net_layers = None
        self.buffer_size = None
        self.run_name = None
        self.num_episodes = None
        self.batch_size = None
        self.gamma = None
        self.critic_lr = None
        self.actor_lr = None
        self.mean = None
        self.stddev = None
        self.episode_to_explore_till = None
        self.head_start_exploration = None
        self.critic_grad_clip = None
        self.actor_grad_clip = None
        self.polyak_const = None
        self.render = None
        self.save_path = None
        self.render_freq = None
        self.save_freq = None

        self.action_dim = 2

        # if variables are set using **kwargs, it would be considered and not the config entry
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise NameError(f"Variable named {k} not defined")
        
        # setting values from config file
        self.configure(self.config)

        # declaring the network
        self.critic = Critic_Transformer(self.input_emb1, self.input_emb2, self.d_model, self.d_k, self.mlp_hidden_layers, self.v_net_layers, self.device).to(self.device)
        # initializing using xavier initialization
        self.critic.apply(self.xavier_init_weights)

        #initializing the fixed targets
        self.critic_target = Critic_Transformer(self.input_emb1, self.input_emb2, self.d_model, self.d_k, self.mlp_hidden_layers, self.v_net_layers, self.device).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # declaring the network
        self.actor = Actor_Transformer(self.input_emb1, self.input_emb2, self.d_model, self.d_k, self.mlp_hidden_layers, self.a_net_layers, self.mean, self.stddev, self.episode_to_explore_till, self.device).to(self.device)
        # initializing using xavier initialization
        self.actor.apply(self.xavier_init_weights)

        #initializing the fixed targets
        self.actor_target = Actor_Transformer(self.input_emb1, self.input_emb2, self.d_model, self.d_k, self.mlp_hidden_layers, self.a_net_layers, self.mean, self.stddev, self.episode_to_explore_till, self.device).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # initalizing the replay buffer
        self.experience_replay = ExperienceReplay(self.buffer_size)

        # variable to keep count of the number of steps that has occured
        self.steps = 0

        if self.run_name is not None:
            self.writer = SummaryWriter('runs/'+self.run_name)
        else:
            self.writer = SummaryWriter()

    def configure(self, config:str):
        with open(config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)

        if self.input_emb1 is None:
            self.input_emb1 = config["input_emb1"]
            assert(self.input_emb1 is not None), f"Argument input_emb1 cannot be None"

        if self.input_emb2 is None:
            self.input_emb2 = config["input_emb2"]
            assert(self.input_emb2 is not None), f"Argument input_emb2 cannot be None"

        if self.d_model is None:
            self.d_model = config["d_model"]
            assert(self.d_model is not None), f"Argument d_model cannot be None"

        if self.d_k is None:
            self.d_k = config["d_k"]
            assert(self.d_k is not None), f"Argument d_k cannot be None"

        if self.mlp_hidden_layers is None:
            self.mlp_hidden_layers = config["mlp_hidden_layers"]
            assert(self.mlp_hidden_layers is not None), f"Argument mlp_hidden_layers cannot be None"

        if self.v_net_layers is None:
            self.v_net_layers = config["v_net_layers"]
            assert(self.v_net_layers is not None), f"Argument v_net_layers cannot be None"

        if self.a_net_layers is None:
            self.a_net_layers = config["a_net_layers"]
            assert(self.a_net_layers is not None), f"Argument a_net_layers cannot be None"

        if self.buffer_size is None:
            self.buffer_size = config["buffer_size"]
            assert(self.buffer_size is not None), f"Argument buffer_size cannot be None"

        if self.num_episodes is None:
            self.num_episodes = config["num_episodes"]
            assert(self.num_episodes is not None), f"Argument num_episodes cannot be None"

        if self.batch_size is None:
            self.batch_size = config["batch_size"]
            assert(self.batch_size is not None), f"Argument batch_size cannot be None"

        if self.gamma is None:
            self.gamma = config["gamma"]
            assert(self.gamma is not None), f"Argument gamma cannot be None"

        if self.critic_lr is None:
            self.critic_lr = config["critic_lr"]
            assert(self.critic_lr is not None), f"Argument critic_lr cannot be None"

        if self.actor_lr is None:
            self.actor_lr = config["actor_lr"]
            assert(self.actor_lr is not None), f"Argument actor_lr cannot be None"

        if self.mean is None:
            self.mean = config["mean"]
            assert(self.mean is not None), f"Argument mean cannot be None"

        if self.stddev is None:
            self.stddev = config["stddev"]
            assert(self.stddev is not None), f"Argument stddev cannot be None"

        if self.episode_to_explore_till is None:
            self.episode_to_explore_till = config["episode_to_explore_till"]
            assert(self.episode_to_explore_till is not None), f"Argument episode_to_explore_till cannot be None"

        if self.head_start_exploration is None:
            self.head_start_exploration = config["head_start_exploration"]
            assert(self.head_start_exploration is not None), f"Argument head_start_exploration cannot be None"

        if self.critic_grad_clip is None:
            self.critic_grad_clip = config["critic_grad_clip"]
            assert(self.critic_grad_clip is not None), f"Argument critic_grad_clip cannot be None"

        if self.actor_grad_clip is None:
            self.actor_grad_clip = config["actor_grad_clip"]
            assert(self.actor_grad_clip is not None), f"Argument actor_grad_clip cannot be None"

        if self.polyak_const is None:
            self.polyak_const = config["polyak_const"]
            assert(self.polyak_const is not None), f"Argument polyak_const cannot be None"

        if self.render is None:
            self.render = config["render"]
            assert(self.render is not None), f"Argument render cannot be None"

        if self.save_path is None:
            self.save_path = config["save_path"]
            assert(self.save_path is not None), f"Argument save_path cannot be None"

        if self.render_freq is None:
            self.render_freq = config["render_freq"]
            assert(self.render_freq is not None), f"Argument render_freq cannot be None"

        if self.save_freq is None:
            self.save_freq = config["save_freq"]
            assert(self.save_freq is not None), f"Argument save_freq cannot be None"
            
    def xavier_init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    
    def preprocess_observation(self, obs):
        """
        To convert dict observation to numpy observation
        """
        assert(type(obs) == dict)
        observation = np.array([], dtype=np.float32)
        observation = np.concatenate((observation, obs["goal"].flatten()) )
        observation = np.concatenate((observation, obs["humans"].flatten()) )
        observation = np.concatenate((observation, obs["laptops"].flatten()) )
        observation = np.concatenate((observation, obs["tables"].flatten()) )
        observation = np.concatenate((observation, obs["plants"].flatten()) )
        if "walls" in obs.keys():
            observation = np.concatenate((observation, obs["walls"].flatten()))
        return observation
    
    def postprocess_observation(self, obs):
        """
        To convert a one-vector observation into two inputs that can be given to the transformer
        """
        if(len(obs.shape) == 1):
            obs = obs.reshape(1, -1)
        
        robot_state = obs[:, 0:self.env.observation_space["goal"].shape[0]].reshape(obs.shape[0], -1, self.env.observation_space["goal"].shape[0])
        entity_state = obs[:, self.env.observation_space["goal"].shape[0]:].reshape(obs.shape[0], -1, 13)
        
        return robot_state, entity_state

    def get_action(self, current_state):
        with torch.no_grad():
            robot_state, entity_state = self.postprocess_observation(current_state)
            action_continuous = self.actor(torch.from_numpy(robot_state).float().to(self.device), torch.from_numpy(entity_state).float().to(self.device))
            return [action_continuous[0].item(), action_continuous[1].item()]
    
    def save_model(self, critic_path, actor_path):
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.actor.state_dict(), actor_path)

    def update(self):
        curr_state, rew, act, next_state, done = self.experience_replay.sample_batch(self.batch_size)
                    
        # getting robot and entity observations from state
        curr_state_robot, curr_state_entity = self.postprocess_observation(curr_state)

        # getting robot and entity observations from next_state
        next_state_robot, next_state_entity = self.postprocess_observation(next_state)

        # converting to pytorch tensor
        curr_state_robot = torch.from_numpy(curr_state_robot).float().to(self.device)
        curr_state_entity = torch.from_numpy(curr_state_entity).float().to(self.device)
        next_state_robot = torch.from_numpy(next_state_robot).float().to(self.device)
        next_state_entity = torch.from_numpy(next_state_entity).float().to(self.device)
        rew = torch.from_numpy(rew).float().to(self.device)
        act = torch.from_numpy(act).float().to(self.device).reshape(-1, act.shape[1], self.action_dim)
        done = torch.from_numpy(done).float().to(self.device)

        # Calculate current state Q(s,a)
        curr_Q = self.critic(curr_state_robot, curr_state_entity, act).squeeze(1)
        next_actions = self.actor_target(next_state_robot, next_state_entity).reshape(-1, act.shape[1], self.action_dim)
        # Calculate next state Q'(s',pi'(s'))
        next_Q = self.critic_target(next_state_robot, next_state_entity, next_actions.detach()).squeeze(1)
        # calculating target value given by r + (gamma * Q(s', a_max, theta')) where theta' is the target network parameters
        # if the transition has done=True, then the target is just r
        expected_Q = rew + (1-done) * self.gamma * next_Q
        # update critic
        q_loss = self.loss_fn(curr_Q, expected_Q)

        self.critic_optimizer.zero_grad()
        q_loss.backward() 
        # gradient clipping
        self.total_critic_grad_norm += torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_grad_clip).item()
        self.critic_optimizer.step()

        self.critic_episode_loss += q_loss.item()

        # update actor
        policy_loss = -self.critic(curr_state_robot, curr_state_entity, self.actor(curr_state_robot, curr_state_entity).reshape(-1, act.shape[1], self.action_dim)).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        # gradient clipping
        self.total_actor_grad_norm += torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_grad_clip).item()
        self.actor_optimizer.step()

        self.actor_episode_loss += policy_loss.item()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.polyak_const + target_param.data * (1.0 - self.polyak_const))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.polyak_const + target_param.data * (1.0 - self.polyak_const))

        

    def plot(self, episode):
        self.rewards.append(self.episode_reward)
        self.critic_losses.append(self.critic_episode_loss/self.batch_size)
        self.actor_losses.append(self.actor_episode_loss/self.batch_size)
        self.critic_grad_norms.append(self.total_critic_grad_norm/self.batch_size)
        self.actor_grad_norms.append(self.total_actor_grad_norm/self.batch_size)
        self.successes.append(self.has_reached_goal)
        self.collisions.append(self.has_collided)
        self.steps_to_reach.append(self.steps)
        self.discomforts_sngnn.append(self.discomfort_sngnn)
        self.discomforts_crowdnav.append(self.discomfort_crowdnav)

        if not os.path.isdir(os.path.join(self.save_path, "plots")):
            os.makedirs(os.path.join(self.save_path, "plots"))

        np.save(os.path.join(self.save_path, "plots", "rewards"), np.array(self.rewards), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "critic_losses"), np.array(self.critic_losses), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "actor_losses"), np.array(self.actor_losses), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "critic_grad_norms"), np.array(self.critic_grad_norms), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "actor_grad_norms"), np.array(self.actor_grad_norms), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "successes"), np.array(self.successes), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "collisions"), np.array(self.collisions), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "steps_to_reach"), np.array(self.steps_to_reach), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "discomfort_sngnn"), np.array(self.discomforts_sngnn), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "discomfort_crowdnav"), np.array(self.discomforts_crowdnav), allow_pickle=True, fix_imports=True)


        self.writer.add_scalar("reward / epsiode", self.episode_reward, episode)
        self.writer.add_scalar("critic loss / episode", self.critic_episode_loss/self.batch_size, episode)
        self.writer.add_scalar("actor loss / episode", self.actor_episode_loss/self.batch_size, episode)
        self.writer.add_scalar("Average critic grad norm / episode", (self.total_critic_grad_norm/self.batch_size), episode)
        self.writer.add_scalar("Average actor grad norm / episode", (self.total_actor_grad_norm/self.batch_size), episode)
        self.writer.add_scalar("ending in sucess? / episode", self.has_reached_goal, episode)
        self.writer.add_scalar("has collided? / episode", self.has_collided, episode)
        self.writer.add_scalar("Steps to reach goal / episode", self.steps, episode)
        self.writer.add_scalar("Discomfort SNGNN / episode", self.discomfort_sngnn, episode)
        self.writer.add_scalar("Discomfort CrowdNav / episode", self.discomfort_crowdnav, episode)
        self.writer.flush()  

    def train(self):
        self.loss_fn = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.rewards = []
        self.critic_losses = []
        self.actor_losses = []
        self.exploration_rates = []
        self.critic_grad_norms = []
        self.actor_grad_norms = []
        self.successes = []
        self.collisions = []
        self.steps_to_reach = []
        self.discomforts_sngnn = []
        self.discomforts_crowdnav = []

        self.average_reward = 0

        # train loop
        for i in range(self.num_episodes):
            current_obs = self.env.reset()
            current_obs = self.preprocess_observation(current_obs)
            done = False
            self.episode_reward = 0
            self.critic_episode_loss = 0
            self.actor_episode_loss = 0
            self.total_critic_grad_norm = 0
            self.total_actor_grad_norm = 0
            self.has_reached_goal = 0
            self.has_collided = 0
            self.steps = 0
            self.discomfort_sngnn = 0
            self.discomfort_crowdnav = 0

            while not done:
                # sampling an action from the current state
                if i < self.head_start_exploration:
                    action_continuous = np.array([np.random.uniform(-1.0,1.0), np.random.uniform(-1.0,1.0)])
                else:
                    action_continuous = self.get_action(current_obs)

                # taking a step in the environment
                next_obs, reward, done, info = self.env.step(action_continuous)

                # incrementing total steps
                self.steps += 1

                # preprocessing the observation, i.e padding the observation with zeros if it is lesser than the maximum size
                next_obs = self.preprocess_observation(next_obs)
                
                # rendering if reqd
                if self.render and ((i+1) % self.render_freq == 0):
                    self.env.render()

                # storing the rewards
                self.episode_reward += reward

                # storing discomforts
                self.discomfort_sngnn += info["DISCOMFORT_SNGNN"]
                self.discomfort_crowdnav += info["DISCOMFORT_CROWDNAV"]

                # storing whether the agent reached the goal
                if info["REACHED_GOAL"]:
                    self.has_reached_goal = 1
                
                if info["COLLISION"]:
                    self.has_collided = 1
                    self.steps = self.env.EPISODE_LENGTH

                # storing the current state transition in the replay buffer. 
                self.experience_replay.insert((current_obs, reward, action_continuous, next_obs, done))

                # sampling a mini-batch of state transitions if the replay buffer has sufficent examples
                if len(self.experience_replay) > self.batch_size:
                    self.update()

                # setting the current observation to the next observation
                current_obs = next_obs
           
            # plotting using tensorboard
            print(f"Episode {i+1} Reward: {round(self.episode_reward,4)}")
            self.plot(i+1)

            # saving model
            if (self.save_path is not None) and ((i+1)%self.save_freq == 0) and self.episode_reward >= self.average_reward:
                if not os.path.isdir(self.save_path):
                    os.makedirs(self.save_path)
                try:
                    self.save_model(os.path.join(self.save_path, "critic_episode"+ str(i+1).zfill(8) + ".pth"), os.path.join(self.save_path, "actor_episode"+ str(i+1).zfill(8) + ".pth"))
                except:
                    print("Error in saving model")

            # updating the average reward
            if (i+1) % self.save_freq == 0:
                self.average_reward = 0
            else:
                self.average_reward = ((i%self.save_freq)*self.average_reward + self.episode_reward)/((i%self.save_freq)+1)
   
    def eval(self, num_episodes, path=None):
        if path is not None:
            self.actor.load_state_dict(torch.load(path, map_location=torch.self.device(self.device)))
        
        self.actor.eval()

        total_reward = 0
        successive_runs = 0
        for i in range(num_episodes):
            o = self.env.reset()
            o = self.preprocess_observation(o)
            done = False
            while not done:
                act_continuous = self.get_action(o, 0)
                new_state, reward, done, info = self.env.step(act_continuous)
                new_state = self.preprocess_observation(new_state)
                total_reward += reward

                self.env.render()

                if info["REACHED_GOAL"]:
                    successive_runs += 1

                o = new_state

        print(f"Total episodes run: {num_episodes}")
        print(f"Total successive runs: {successive_runs}")
        print(f"Average reward per episode: {total_reward/num_episodes}")

if __name__ == "__main__":
    env = gym.make("SocNavEnv-v1")
    env.configure("./configs/env.yaml")
    env.set_padded_observations(False)
    robot_state_dim = env.observation_space["goal"].shape[0]
    entity_state_dim = 13
    config = "./configs/ddpg_transformer.yaml"
    agent = DDPG_Transformer_Agent(env, config, input_emb1=robot_state_dim, input_emb2=entity_state_dim, run_name="ddpg_transformer_SocNavEnv")
    agent.train()