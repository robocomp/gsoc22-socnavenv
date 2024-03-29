import torch
import torch.nn as nn
import torch.nn.functional as F
import socnavgym
import gym
import numpy as np
import os
import torch.optim as optim
import argparse
import yaml
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from agents.models import MLP

class ActorCritic(nn.Module):
    def __init__(self, input_dim:int, policy_net_hidden_layers:list, value_net_hidden_layers:list):
        super(ActorCritic, self).__init__()
        self.policy_net = MLP(input_dim, policy_net_hidden_layers)
        self.value_net = MLP(input_dim, value_net_hidden_layers)
        
    def forward(self, state):
        logits = self.policy_net(state)
        value = self.value_net(state)
        return logits, value

class A2CAgent:
    def __init__(self, env:gym.Env, config:str, **kwargs):
        
        # initializing the env
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        # agent variables
        self.input_layer_size = None
        self.policy_net_hidden_layers = None
        self.value_net_hidden_layers = None
        self.num_episodes = None
        self.gamma = None
        self.lr = None
        self.entropy_penalty = None
        self.save_path = None
        self.render = None
        self.render_freq = None
        self.save_freq = None
        self.run_name = None

        # if variables are set using **kwargs, it would be considered and not the config entry
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise NameError(f"Variable named {k} not defined")
        
        # setting values from config file
        self.configure(self.config)

        # initializing model
        self.model = ActorCritic(self.input_layer_size, self.policy_net_hidden_layers, self.value_net_hidden_layers).to(self.device)

        # variable to keep count of the number of steps that has occured
        self.steps = 0

        # tensorboard run directory
        if self.run_name is not None:
            self.writer = SummaryWriter('runs/'+self.run_name)
        else:
            self.writer = SummaryWriter()

    def configure(self, config:str):
        with open(config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)
        
        if self.input_layer_size is None:
             self.input_layer_size = config["input_layer_size"]
             assert(self.input_layer_size is not None), "Argument input_layer_size cannot be None"

        if self.policy_net_hidden_layers is None:
             self.policy_net_hidden_layers = config["policy_net_hidden_layers"]
             assert(self.policy_net_hidden_layers is not None), "Argument policy_net_hidden_layers cannot be None"

        if self.value_net_hidden_layers is None:
             self.value_net_hidden_layers = config["value_net_hidden_layers"]
             assert(self.value_net_hidden_layers is not None), "Argument value_net_hidden_layers cannot be None"

        if self.num_episodes is None:
             self.num_episodes = config["num_episodes"]
             assert(self.num_episodes is not None), "Argument num_episodes cannot be None"

        if self.gamma is None:
             self.gamma = config["gamma"]
             assert(self.gamma is not None), "Argument gamma cannot be None"

        if self.lr is None:
             self.lr = config["lr"]
             assert(self.lr is not None), "Argument lr cannot be None"

        if self.entropy_penalty is None:
             self.entropy_penalty = config["entropy_penalty"]
             assert(self.entropy_penalty is not None), "Argument entropy_penalty cannot be None"

        if self.save_path is None:
             self.save_path = config["save_path"]
             assert(self.save_path is not None), "Argument save_path cannot be None"

        if self.render is None:
             self.render = config["render"]
             assert(self.render is not None), "Argument render cannot be None"

        if self.render_freq is None:
             self.render_freq = config["render_freq"]
             assert(self.render_freq is not None), "Argument render_freq cannot be None"

        if self.save_freq is None:
             self.save_freq = config["save_freq"]
             assert(self.save_freq is not None), "Argument save_freq cannot be None"



    def xavier_init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    def preprocess_observation(self, obs):
        """
        To convert dict observation to numpy observation
        """
        assert(type(obs) == dict)
        observation = np.array([], dtype=np.float32)
        if "goal" in obs.keys() : observation = np.concatenate((observation, obs["goal"].flatten()) )
        if "humans" in obs.keys() : observation = np.concatenate((observation, obs["humans"].flatten()) )
        if "laptops" in obs.keys() : observation = np.concatenate((observation, obs["laptops"].flatten()) )
        if "tables" in obs.keys() : observation = np.concatenate((observation, obs["tables"].flatten()) )
        if "plants" in obs.keys() : observation = np.concatenate((observation, obs["plants"].flatten()) )
        if "walls" in obs.keys():
            observation = np.concatenate((observation, obs["walls"].flatten()))
        return observation


    def get_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            logits, _ = self.model.forward(state)
            dist = F.softmax(logits, dim=0)
            probs = Categorical(dist)
            return probs.sample().cpu().detach().item()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def update(self):
        states = torch.FloatTensor(np.array([sars[0] for sars in self.trajectory])).to(self.device)
        actions = torch.LongTensor(np.array([sars[1] for sars in self.trajectory])).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(np.array([sars[2] for sars in self.trajectory])).to(self.device)
        next_states = torch.FloatTensor(np.array([sars[3] for sars in self.trajectory])).to(self.device)
        dones = torch.FloatTensor(np.array([sars[4] for sars in self.trajectory])).view(-1, 1).to(self.device)
            
        # compute discounted rewards
        discounted_rewards = [torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[j:].size(0))])\
            * rewards[j:]) for j in range(rewards.size(0))]  # sorry, not the most readable code.
        value_targets = rewards.view(-1, 1) + torch.FloatTensor(discounted_rewards).view(-1, 1).to(self.device)
        
        logits, values = self.model.forward(states)
        dists = F.softmax(logits, dim=1)
        probs = Categorical(dists)
        
        # compute value loss
        value_loss = F.mse_loss(values, value_targets.detach())
        
        # compute entropy bonus
        entropy = -torch.mean(torch.sum(dists * torch.log(torch.clamp(dists, 1e-10,1.0)), dim=1))

        # compute policy loss
        advantage = value_targets - values
        policy_loss = -probs.log_prob(actions.view(actions.size(0))).view(-1, 1) * advantage.detach()
        policy_loss = policy_loss.mean()

        # total loss
        loss = policy_loss + value_loss - self.entropy_penalty*entropy
        self.episode_loss += loss.item()

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        self.total_grad_norm += torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

    def plot(self, episode):
        self.rewards.append(self.episode_reward)
        self.losses.append(self.episode_loss)
        self.grad_norms.append(self.total_grad_norm)
        self.successes.append(self.has_reached_goal)
        self.collisions.append(self.has_collided)
        self.steps_to_reach.append(self.steps)

        if not os.path.isdir(os.path.join(self.save_path, "plots")):
            os.makedirs(os.path.join(self.save_path, "plots"))

        np.save(os.path.join(self.save_path, "plots", "rewards"), np.array(self.rewards), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "losses"), np.array(self.episode_loss), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "grad_norms"), np.array(self.total_grad_norm), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "successes"), np.array(self.has_reached_goal), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "collisions"), np.array(self.has_collided), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "steps_to_reach"), np.array(self.steps), allow_pickle=True, fix_imports=True)

        self.writer.add_scalar("reward / epsiode", self.episode_reward, episode)
        self.writer.add_scalar("loss / episode", self.episode_loss, episode)
        self.writer.add_scalar("total grad norm / episode", (self.total_grad_norm), episode)
        self.writer.add_scalar("ending in sucess? / episode", self.has_reached_goal, episode)
        self.writer.add_scalar("has collided? / episode", self.has_collided, episode)
        self.writer.add_scalar("Steps to reach goal / episode", self.steps, episode)
        self.writer.flush()


    def train(self):

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.rewards = []
        self.losses = []
        self.grad_norms = []
        self.successes = []
        self.collisions = []
        self.steps_to_reach = []
        
        self.average_reward = 0

        for i in range(self.num_episodes):
            # resetting the environment before the episode starts
            current_state, _ = self.env.reset()
            
            # preprocessing the observation
            current_state = self.preprocess_observation(current_state)

            # initializing episode related variables
            done = False
            self.episode_reward = 0
            self.total_grad_norm = 0
            self.episode_loss = 0
            self.has_reached_goal = 0
            self.has_collided = 0
            self.steps = 0

            self.trajectory = [] # [[s, a, r, s', done], [], ...]
            
            while not done:
                action = self.get_action(current_state)
                action_continuous = self.env.discrete_to_continuous_action(action)
                next_state, reward, terminated, truncated, info = self.env.step(action_continuous)
                done = terminated or truncated
                next_state = self.preprocess_observation(next_state)
                self.trajectory.append([current_state, action, reward, next_state, done])
                self.episode_reward += reward
                current_state = next_state
                self.steps += 1
                if info["REACHED_GOAL"]:
                    self.has_reached_goal = 1
                
                if info["COLLISION"]:
                    self.has_collided = 1
                    self.steps = self.env.EPISODE_LENGTH-1
                
                # rendering if reqd
                if self.render and ((i+1) % self.render_freq == 0):
                    self.env.render()
            
            self.update()

            # plotting using tensorboard
            print(f"Episode {i+1} Reward: {self.episode_reward} Loss: {self.episode_loss}")
            self.plot(i+1)

            # saving model
            if (self.save_path is not None) and ((i+1)%self.save_freq == 0) and self.episode_reward >= self.average_reward:
                if not os.path.isdir(self.save_path):
                    os.makedirs(self.save_path)
                try:
                    self.save_model(os.path.join(self.save_path, "episode"+ str(i+1).zfill(8) + ".pth"))
                except:
                    print("Error in saving model")

            # updating the average reward
            if (i+1) % self.save_freq == 0:
                self.average_reward = 0
            else:
                self.average_reward = ((i%self.save_freq)*self.average_reward + self.episode_reward)/((i%self.save_freq)+1)

    def eval(self, num_episodes, path=None):
        # intialising metrics
        discomfort_sngnn = 0
        discomfort_crowdnav = 0
        timeout = 0
        success_rate = 0
        time_taken = 0
        closest_human_dist = 0
        closest_obstacle_dist = 0
        collision_rate = 0

        print("Loading Model")
        if path is not None:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        
        self.model.eval()

        print("done")
        from tqdm import tqdm
        total_reward = 0

        print(f"Evaluating model for {num_episodes} episodes")
        for i in tqdm(range(num_episodes)):
            o, _ = self.env.reset()
            o = self.preprocess_observation(o)
            done = False
            episode_reward = 0
            has_reached_goal = 0
            has_collided = 0
            has_timed_out = 0
            steps = 0
            episode_discomfort_sngnn = 0
            episode_discomfort_crowdnav = 0
            min_human_dist = float('inf')
            min_obstacle_dist = float('inf')
            
            while not done:
                action = self.get_action(o)
                act_continuous = self.env.discrete_to_continuous_action(action)
                new_state, reward, terminated, truncated, info = self.env.step(act_continuous)
                done = terminated or truncated
                new_state = self.preprocess_observation(new_state)
                total_reward += reward

                # self.env.render()
                steps += 1

                # storing the rewards
                episode_reward += reward

                # storing discomforts
                episode_discomfort_sngnn += info["sngnn_reward"]
                episode_discomfort_crowdnav += info["DISCOMFORT_CROWDNAV"]

                # storing whether the agent reached the goal
                if info["REACHED_GOAL"]:
                    has_reached_goal = 1
                
                if info["COLLISION"]:
                    has_collided = 1
                    steps = self.env.EPISODE_LENGTH
                
                if info["MAX_STEPS"]:
                    has_timed_out = 1

                min_human_dist = min(min_human_dist, info["closest_human_dist"])
                min_obstacle_dist = min(min_obstacle_dist, info["closest_obstacle_dist"])

                episode_reward += reward

                o = new_state

            discomfort_sngnn += episode_discomfort_sngnn
            discomfort_crowdnav += episode_discomfort_crowdnav
            timeout += has_timed_out
            success_rate += has_reached_goal
            time_taken += steps
            closest_human_dist += min_human_dist
            closest_obstacle_dist += min_obstacle_dist
            collision_rate += has_collided
        
        print(f"Average discomfort_sngnn: {discomfort_sngnn/num_episodes}") 
        print(f"Average discomfort_crowdnav: {discomfort_crowdnav/num_episodes}") 
        print(f"Average timeout: {timeout/num_episodes}") 
        print(f"Average success_rate: {success_rate/num_episodes}") 
        print(f"Average time_taken: {time_taken/num_episodes}") 
        print(f"Average closest_human_dist: {closest_human_dist/num_episodes}") 
        print(f"Average closest_obstacle_dist: {closest_obstacle_dist/num_episodes}") 
        print(f"Average collision_rate: {collision_rate/num_episodes}") 
