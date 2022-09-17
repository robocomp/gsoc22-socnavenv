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
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from agents.models import MLP, Transformer

class A2C_Transformer(nn.Module):
    def __init__(
        self,
        input_emb1:int,
        input_emb2:int,
        d_model:int,
        d_k:int,
        actor_mlp_hidden_layers:list,
        critic_mlp_hidden_layers:list,
    ) -> None:
        
        super(A2C_Transformer, self).__init__()
        self.transformer = Transformer(input_emb1, input_emb2, d_model, d_k, None)
        self.policy_net = MLP(2*d_model, actor_mlp_hidden_layers)
        self.value_net = MLP(2*d_model, critic_mlp_hidden_layers)

    def forward(self, inp1, inp2):
        x = self.transformer(inp1, inp2)
        logits = self.policy_net(x)
        value = self.value_net(x)
        return logits, value


class A2C_Transformer_Agent:
    def __init__(self, env:gym.Env, config:str, **kwargs):
        assert(env is not None and config is not None)
        
        # agent variables
        self.input_emb1 = None
        self.input_emb2 = None
        self.d_model = None
        self.d_k = None
        self.actor_mlp_hidden_layers = None
        self.critic_mlp_hidden_layers = None
        self.num_episodes = None
        self.gamma = None
        self.lr = None
        self.entropy_penalty = None
        self.save_path = None
        self.render = None
        self.render_freq = None
        self.save_freq = None
        self.run_name = None

        # initializing the environment
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        # if variables are set using **kwargs, it would be considered and not the config entry
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise NameError(f"Variable named {k} not defined")
        
        # setting values from config file
        self.configure(self.config)

        # initializing the model
        self.model = A2C_Transformer(
            self.input_emb1,
            self.input_emb2,
            self.d_model,
            self.d_k,
            self.actor_mlp_hidden_layers,
            self.critic_mlp_hidden_layers
        ).to(self.device)

        # initializing model weights using xavier initialisation
        self.model.apply(self.xavier_init_weights)

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

        if self.lr is None:
            self.lr = config["lr"]
            assert(self.lr is not None), "self cannot be None.lr"

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
            logits, _ = self.model(robot_state, entity_state)
            logits = logits.squeeze()
            dist = F.softmax(logits, dim=0)
            probs = Categorical(dist)
            return probs.sample().cpu().detach().item()

    def update(self):
        states = np.array([sars[0] for sars in self.trajectory])
        # states.shape = (ep_len, 8 + 13*num_entities)

        actions = torch.LongTensor(np.array([sars[1] for sars in self.trajectory])).view(-1, 1).to(self.device)
        # actions.shape = (ep_len, 1)
        
        rewards = torch.FloatTensor(np.array([sars[2] for sars in self.trajectory])).to(self.device)
        # rewards.shape = (ep_len, 1)
        
        next_states = np.array([sars[3] for sars in self.trajectory])
        # next_states.shape = (ep_len, 8 + 13*num_entities)

        dones = torch.FloatTensor(np.array([sars[4] for sars in self.trajectory])).view(-1, 1).to(self.device)
        # dones.shape = (ep_len, 1)

        robot_state, entity_state = self.postprocess_observation(states)
        robot_state = torch.FloatTensor(robot_state).to(self.device)
        entity_state = torch.FloatTensor(entity_state).to(self.device)
        # robot_state.shape = (ep_len, 1, 8) 
        # entity_state.shape = (ep_len, num_entities, 13)

        robot_state_new, entity_state_new = self.postprocess_observation(next_states)
        robot_state_new = torch.FloatTensor(robot_state_new).to(self.device)
        entity_state_new = torch.FloatTensor(entity_state_new).to(self.device)
        # robot_state_new.shape = (ep_len, 1, 8) 
        # entity_state_new.shape = (ep_len, num_entities, 13)

        # compute discounted rewards
        discounted_rewards = [torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[j:].size(0))]).to(self.device)\
            * rewards[j:]) for j in range(rewards.size(0))]
        
        value_targets = rewards.view(-1, 1) + torch.FloatTensor(discounted_rewards).view(-1, 1).to(self.device)
        # value_targets.shape = (ep_len, 1)

        logits, values = self.model(robot_state, entity_state)
        logits = logits.squeeze(1)
        values = values.squeeze(1)
        # logits.shape = (ep_len, a_dim)
        # values.shape = (ep_len, 1)

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
        self.total_grad_norm += torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5).item()

        # gradient descent update
        self.optimizer.step()

    def plot(self, episode):
        self.rewards.append(self.episode_reward)
        self.losses.append(self.episode_loss)
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
        np.save(os.path.join(self.save_path, "plots", "grad_norms"), np.array(self.grad_norms), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "successes"), np.array(self.successes), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "collisions"), np.array(self.collisions), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "steps_to_reach"), np.array(self.steps_to_reach), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "discomfort_sngnn"), np.array(self.discomforts_sngnn), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "discomfort_crowdnav"), np.array(self.discomforts_crowdnav), allow_pickle=True, fix_imports=True)

        self.writer.add_scalar("reward / epsiode", self.episode_reward, episode)
        self.writer.add_scalar("loss / episode", self.episode_loss, episode)
        self.writer.add_scalar("total grad norm / episode", (self.total_grad_norm), episode)
        self.writer.add_scalar("ending in sucess? / episode", self.has_reached_goal, episode)
        self.writer.add_scalar("has collided? / episode", self.has_collided, episode)
        self.writer.add_scalar("Steps to reach goal / episode", self.steps, episode)
        self.writer.add_scalar("Discomfort SNGNN / episode", self.discomfort_sngnn, episode)
        self.writer.add_scalar("Discomfort CrowdNav / episode", self.discomfort_crowdnav, episode)
        self.writer.flush()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)   

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

        for i in range(self.num_episodes):
            # resetting the environment before the episode starts
            current_state = self.env.reset()
            
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
            self.discomfort_sngnn = 0
            self.discomfort_crowdnav = 0

            self.trajectory = [] # [[s, a, r, s', done], [], ...]
            
            while not done:
                action = self.get_action(current_state)
                action_continuous = self.discrete_to_continuous_action(action)
                next_state, reward, done, info = self.env.step(action_continuous)
                next_state = self.preprocess_observation(next_state)
                self.trajectory.append([current_state, action, reward, next_state, done])
                self.episode_reward += reward
                current_state = next_state
                self.steps += 1

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
        if path is not None:
            self.model.load_state_dict(torch.load(path, map_location=torch.self.device(self.device)))
        
        self.model.eval()

        total_reward = 0
        successive_runs = 0
        for i in range(num_episodes):
            o = self.env.reset()
            o = self.preprocess_observation(o)
            done = False
            while not done:
                act_discrete = self.get_action(o, 0)
                act_continuous = self.discrete_to_continuous_action(act_discrete)
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

    config = "./configs/a2c_transformer.yaml"
    robot_state_dim = env.observation_space["goal"].shape[0]
    entity_state_dim = 13

    agent = A2C_Transformer_Agent(
        env, 
        config, 
        actor_input_emb1=robot_state_dim,
        actor_input_emb2=entity_state_dim,
        critic_input_emb1=robot_state_dim,
        critic_input_emb2=entity_state_dim,
        run_name="a2c_transformer_SocNavEnv"
    )
    agent.train()