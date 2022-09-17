import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import socnavenv
import gym
import numpy as np
import copy
import os
import random
import torch.optim as optim
import argparse
import yaml
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from agents.models import MLP, RolloutBuffer, Transformer

class PPO_Transformer(nn.Module):
    def __init__(
        self,
        input_emb1:int,
        input_emb2:int,
        d_model:int,
        d_k:int,
        actor_mlp_hidden_layers:list,
        critic_mlp_hidden_layers:list,
    ) -> None:
        super(PPO_Transformer, self).__init__()
        self.transformer = Transformer(input_emb1, input_emb2, d_model, d_k, None)
        self.actor = nn.Sequential(
            MLP(2*d_model, actor_mlp_hidden_layers),
            nn.Softmax(dim=-1)
        )
        self.critic = MLP(2*d_model, critic_mlp_hidden_layers)

    def act(self, inp1, inp2):
        x = self.transformer(inp1, inp2).squeeze(1)
        action_probs = self.actor(x)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach().cpu().item(), action_logprob.detach()
        
    def forward(self, inp1, inp2, action):
        state = self.transformer(inp1, inp2).squeeze(1)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO_Transformer_Agent:
    def __init__(self, env:gym.Env, config:str, **kwargs) -> None:
        assert(env is not None and config is not None)

        # initializing environment
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        # agent variables
        self.input_emb1 = None
        self.input_emb2 = None
        self.d_model = None
        self.d_k = None
        self.actor_mlp_hidden_layers = None
        self.critic_mlp_hidden_layers = None
        self.gamma = None
        self.gae_lambda = None
        self.entropy_pen = None
        self.n_epochs = None
        self.ppo_update_freq = None
        self.policy_clip = None
        self.num_episodes = None
        self.run_name = None
        self.lr = None
        self.render = None
        self.render_freq = None
        self.save_path = None
        self.save_freq = None
        
        # if variables are set using **kwargs, it would be considered and not the config entry
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise NameError(f"Variable named {k} not defined")

        self.configure(self.config)

        # initializing model
        self.model = PPO_Transformer(self.input_emb1, self.input_emb2, self.d_model, self.d_k, self.actor_mlp_hidden_layers, self.critic_mlp_hidden_layers).to(self.device)

        # initializing model weights using xavier initialisation
        self.model.apply(self.xavier_init_weights)

        # old model
        self.old_model = PPO_Transformer(self.input_emb1, self.input_emb2, self.d_model, self.d_k, self.actor_mlp_hidden_layers, self.critic_mlp_hidden_layers).to(self.device)

        # removing the old model from the computation graph
        for params in self.old_model.parameters():
            params.requires_grad = False
        
        # initializing with same weights
        self.old_model.load_state_dict(self.model.state_dict())

        # initializing buffer
        self.buffer = RolloutBuffer()

        # tensorboard run directory
        if self.run_name is not None:
            self.writer = SummaryWriter('runs/'+self.run_name)
        else:
            self.writer = SummaryWriter() 

    def configure(self, config:str):
        with open(config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)

        if self.input_emb1 is None:
            self.input_emb1 = config["input_emb1"]
            assert(self.input_emb1 is not None), "Argument input_emb1 cannot be None"
            
        if self.input_emb2 is None:
            self.input_emb2 = config["input_emb2"]
            assert(self.input_emb2 is not None), "Argument input_emb2 cannot be None"

        if self.d_model is None:
            self.d_model = config["d_model"]
            assert(self.d_model is not None), "Argument d_model cannot be None"

        if self.d_k is None:
            self.d_k = config["d_k"]
            assert(self.d_k is not None), "Argument d_k cannot be None"

        if self.actor_mlp_hidden_layers is None:
            self.actor_mlp_hidden_layers = config["actor_mlp_hidden_layers"]
            assert(self.actor_mlp_hidden_layers is not None), "Argument actor_mlp_hidden_layers cannot be None"

        if self.critic_mlp_hidden_layers is None:
            self.critic_mlp_hidden_layers = config["critic_mlp_hidden_layers"]
            assert(self.critic_mlp_hidden_layers is not None), "Argument critic_mlp_hidden_layers cannot be None"
        
        if self.num_episodes is None:
            self.num_episodes = config["num_episodes"]
            assert(self.num_episodes is not None), "Argument num_episodes cannot be None"

        if self.gamma is None:
            self.gamma = config["gamma"]
            assert(self.gamma is not None), "Argument gamma cannot be None"

        if self.gae_lambda is None:
            self.gae_lambda = config["gae_lambda"]
            assert(self.gae_lambda is not None), "Argument gae_lambda cannot be None"

        if self.entropy_pen is None:
            self.entropy_pen = config["entropy_pen"]
            assert(self.entropy_pen is not None), "Argument entropy_pen cannot be None"

        if self.n_epochs is None:
            self.n_epochs = config["n_epochs"]
            assert(self.n_epochs is not None), "Argument n_epochs cannot be None"

        if self.ppo_update_freq is None:
            self.ppo_update_freq = config["ppo_update_freq"]
            assert(self.ppo_update_freq is not None), "Argument ppo_update_freq cannot be None"

        if self.policy_clip is None:
            self.policy_clip = config["policy_clip"]
            assert(self.policy_clip is not None), "Argument policy_clip cannot be None"

        if self.lr is None:
            self.lr = config["lr"]
            assert(self.lr is not None), "Argument lr cannot be None"
            
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

    def get_action(self, state):
        with torch.no_grad():
            robot_state, entity_state = self.postprocess_observation(state)
            robot_state = torch.FloatTensor(robot_state).to(self.device)
            entity_state = torch.FloatTensor(entity_state).to(self.device)
            action, action_logprob = self.old_model.act(robot_state, entity_state)
            self.buffer.logprobs.append(action_logprob.cpu())
            return action

    def calculate_deltas(self, values, rewards, dones):
        deltas = []
        next_value = 0
        rewards = rewards.unsqueeze(-1) # shape = (num_steps, 1)
        dones = dones.unsqueeze(-1) # shape = (num_steps, 1)
        masks = 1-dones # shape = (num_steps, 1)
        for t in reversed(range(0, len(rewards))):
            td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
            next_value = values.data[t]
            deltas.insert(0,td_error)
        deltas = torch.stack(deltas) # shape = (num_steps, 1)

        return deltas

    def calculate_returns(self,rewards, discount_factor):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + R * discount_factor
            returns.insert(0, R)
        
        returns_tensor = torch.stack(returns).to(self.device)
        return returns_tensor


    def calculate_advantages(self, values, rewards, dones):
        advantages = []
        next_value = 0
        advantage = 0
        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1)
        masks = 1 - dones
        for t in reversed(range(0, len(rewards))):
            td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
            next_value = values.data[t]
            
            advantage = td_error + (self.gamma * self.gae_lambda * advantage * masks[t])
            advantages.insert(0, advantage)
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        return advantages
    

    def update(self):
        old_states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        old_actions = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).to(self.device)
        rewards = torch.FloatTensor(np.array(self.buffer.rewards)).to(self.device)
        dones = torch.FloatTensor(np.array(self.buffer.dones)).long().to(self.device)

        state_value_target = self.calculate_returns(rewards, self.gamma).detach().unsqueeze(-1)

        if len(self.buffer.states) != 1:
            for _ in range(self.n_epochs):
                old_robot_states, old_entity_states = self.postprocess_observation(old_states)
                logprobs, state_values, entropy = self.model(old_robot_states, old_entity_states, old_actions)
                advantage = self.calculate_advantages(state_values, rewards, dones).to(self.device)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs)
                # Finding Surrogate Loss
                surr1 = ratios * advantage.detach()
                surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * advantage.detach()

                # final loss of clipped objective PPO
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_pen*(entropy.mean())
                critic_loss = F.mse_loss(state_values, state_value_target)
                
                loss = policy_loss + critic_loss
                self.episode_loss += loss.item()
                loss.backward()

                # gradient clipping
                self.total_grad_norm += torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5).item()
                self.optimizer.step()

        self.old_model.load_state_dict(self.model.state_dict())
        self.buffer.clear()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def plot(self, episode):
        self.rewards.append(self.episode_reward)
        self.losses.append(self.episode_loss/self.n_epochs)
        self.grad_norms.append(self.total_grad_norm/self.n_epochs)
        self.successes.append(self.has_reached_goal)
        self.collisions.append(self.has_collided)
        self.steps_to_reach.append(self.steps)
        self.discomforts_sngnn.append(self.discomfort_sngnn)
        self.discomforts_crowdnav.append(self.discomfort_crowdnav)


        if not os.path.isdir(os.path.join(self.save_path, "plots")):
            os.makedirs(os.path.join(self.save_path, "plots"))

        np.save(os.path.join(self.save_path, "plots", "rewards"), np.array(self.rewards), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "losses"), np.array(self.losses), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "grad_norms"), np.array(self.grad_norms), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "successes"), np.array(self.successes), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "collisions"), np.array(self.collisions), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "steps_to_reach"), np.array(self.steps_to_reach), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "discomfort_sngnn"), np.array(self.discomforts_sngnn), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "discomfort_crowdnav"), np.array(self.discomforts_crowdnav), allow_pickle=True, fix_imports=True)

        self.writer.add_scalar("reward / epsiode", self.episode_reward, episode)
        self.writer.add_scalar("avg loss / episode", self.episode_loss/self.n_epochs, episode)
        self.writer.add_scalar("average total grad norm / episode", (self.total_grad_norm/self.n_epochs), episode)
        self.writer.add_scalar("ending in sucess? / episode", self.has_reached_goal, episode)
        self.writer.add_scalar("has collided? / episode", self.has_collided, episode)
        self.writer.add_scalar("Steps to reach goal / episode", self.steps, episode)
        self.writer.add_scalar("Discomfort SNGNN / episode", self.discomfort_sngnn, episode)
        self.writer.add_scalar("Discomfort CrowdNav / episode", self.discomfort_crowdnav, episode)
        self.writer.flush()

    def train(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.rewards = []
        self.losses = []
        self.grad_norms = []
        self.successes = []
        self.collisions = []
        self.steps_to_reach = []
        self.discomforts_sngnn = []
        self.discomforts_crowdnav = []

        self.average_reward = 0

        # initialize train related parameters
        for i in range(self.num_episodes):
            self.episode_reward = 0
            self.total_grad_norm = 0
            self.episode_loss = 0
            self.has_reached_goal = 0
            self.has_collided = 0
            self.steps = 0
            self.discomfort_sngnn = 0
            self.discomfort_crowdnav = 0

            # resetting the environment before the episode starts
            current_state = self.env.reset()

            # preprocessing the observation
            current_state = self.preprocess_observation(current_state)

            done  = False
            while not done:
                action = self.get_action(current_state)
                action_continuous = self.discrete_to_continuous_action(action)
                next_state, reward, done, info = self.env.step(action_continuous)
                next_state = self.preprocess_observation(next_state)

                self.buffer.states.append(current_state)
                self.buffer.actions.append(action)
                self.buffer.dones.append(done)
                self.buffer.rewards.append(reward)

                current_state = next_state
                
                self.steps += 1
                self.episode_reward += reward

                # storing discomforts
                self.discomfort_sngnn += info["DISCOMFORT_SNGNN"]
                self.discomfort_crowdnav += info["DISCOMFORT_CROWDNAV"]

                
                if info["REACHED_GOAL"]:
                    self.has_reached_goal = 1
                
                if info["COLLISION"]:
                    self.has_collided = 1
                    self.steps = self.env.EPISODE_LENGTH
                
                # rendering if reqd
                if self.render and ((i+1) % self.render_freq == 0):
                    self.env.render()

            if i % self.ppo_update_freq == 0:
                self.update()
            print(f"Episode {i+1} Reward: {self.episode_reward} Loss: {self.episode_loss/self.n_epochs}")
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
        if path is not None:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        
        self.model.eval()

        total_reward = 0
        successive_runs = 0
        for i in range(num_episodes):
            o = self.env.reset()
            o = self.preprocess_observation(o)
            done = False
            while not done:
                act_continuous = self.discrete_to_continuous_action(self.get_action(o))
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
    env.set_padded_observations(True)
    agent = PPO_Transformer_Agent(env, config="./configs/ppo_transformer.yaml")
    agent.train()

