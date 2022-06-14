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
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

###### HYPERPARAMETERS############
LR = 0.001
BUFFER_SIZE = 200000
BATCH_SIZE = 32
GAMMA = 0.99
NUM_EPISODES = 100000
EPSILON = 1
POLYAK_CONSTANT = 0.995
##################################

class MLP(nn.Module):
    def __init__(self, input_layer_size:int, hidden_layers:list, last_relu=False) -> None:
        super().__init__()
        self.layers = []
        self.layers.append(nn.Linear(input_layer_size, hidden_layers[0]))
        self.layers.append(nn.LeakyReLU())
        for i in range(len(hidden_layers)-1):
            if i != (len(hidden_layers)-2):
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                self.layers.append(nn.LeakyReLU())
            elif last_relu:
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                self.layers.append(nn.LeakyReLU())
            else:
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.network = nn.Sequential(*self.layers)
    
    def forward(self, x):
        x = self.network(x)
        return x

class ExperienceReplay:
    def __init__(self, max_capacity) -> None:
        self.list = deque(maxlen = max_capacity)

    def insert(self, val) -> None:
        self.list.append(val)

    def __len__(self):
        return len(self.list)

    def sample_batch(self, batch_size:int):
        sample = random.sample(self.list, batch_size)
        current_state, reward, action, next_state, done = zip(*sample)
        
        current_state = np.array(current_state)
        reward = np.array(reward).reshape(-1, 1)
        action = np.array(action).reshape(-1, 1)
        next_state = np.array(next_state)
        done = np.array(done).reshape(-1, 1)
        
        return current_state, reward, action, next_state, done 

class DQN(nn.Module):
    def __init__(self, input_layer_size:int, hidden_layers:list):
        super(DQN, self).__init__() 
        self.linear_stack = MLP(input_layer_size, hidden_layers)
        
    def forward(self, x):
        x = self.linear_stack(x)
        return x    

class DQNAgent:
    def __init__(self, input_layer_size, hidden_layers, max_capacity, env) -> None:
        # initializing the env
        self.env = env
        
        # declaring the network
        self.model = DQN(input_layer_size, hidden_layers).to(device)
        # initializing weights using xavier initialization
        self.model.apply(self.xavier_init_weights)
        
        #initializing the fixed targets
        self.fixed_targets = DQN(input_layer_size, hidden_layers).to(device)
        self.fixed_targets.load_state_dict(self.model.state_dict())

        # initalizing the replay buffer
        self.experience_replay = ExperienceReplay(int(max_capacity))

        # variable to keep count of the number of steps that has occured
        self.steps = 0

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()
        self.gamma = GAMMA
        self.total_reward = 0

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
        return observation
    
    
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

    def get_action(self, current_state, epsilon):
        
        if np.random.random() > epsilon:
            # exploit
            with torch.no_grad():
                q = self.model(torch.from_numpy(current_state).reshape(1, -1).float().to(device))
                action_discrete = torch.argmax(q).item()
                action_continuous = self.discrete_to_continuous_action(action_discrete)
                return action_continuous, action_discrete

        else:
            # explore
            act = np.random.randint(0, 6)
            return self.discrete_to_continuous_action(act), act
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


    def calculate_grad_norm(self):
        total_norm = 0
        parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def train(
        self,
        num_episodes=NUM_EPISODES,
        epsilon=EPSILON,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        lr = LR,
        polyak_const=POLYAK_CONSTANT,
        render=False,
        save_path = "./models/dqn",
        render_freq = 500,
        save_freq = 500
    ):
        total_reward = 0
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        prev_steps = 0
        
        for i in range(num_episodes):
            # resetting the environment before the episode starts
            current_state = self.env.reset()
            
            # preprocessing the observation
            current_state = self.preprocess_observation(current_state)

            # initializing episode related variables
            done = False
            episode_reward = 0
            episode_loss = 0
            total_grad_norm = 0
            has_reached_goal = False

            while not done:
                # sampling an action from the current state
                action_continuous, action_discrete = self.get_action(current_state, epsilon)
                
                # taking a step in the environment
                next_obs, reward, done, _ = self.env.step(action_continuous)

                # incrementing total steps
                self.steps += 1

                # preprocessing the observation, i.e padding the observation with zeros if it is lesser than the maximum size
                next_obs = self.preprocess_observation(next_obs)
                
                # rendering if reqd
                if render and ((i+1) % render_freq == 0):
                    self.env.render()

                # storing the rewards
                episode_reward += reward

                # storing whether the agent reached the goal
                if reward == 1 and done == True:
                    has_reached_goal = True

                # storing the current state transition in the replay buffer. 
                self.experience_replay.insert((current_state, reward, action_discrete, next_obs, done))

                
                if len(self.experience_replay) > batch_size:
                    # sampling mini-batch from experience replay
                    curr_state, rew, act, next_state, d = self.experience_replay.sample_batch(batch_size)
                    fixed_target_value = torch.max(self.fixed_targets(torch.from_numpy(next_state).float().to(device)), dim=1, keepdim=True).values
                    fixed_target_value = fixed_target_value * (~torch.from_numpy(d).bool().to(device))
                    target = torch.from_numpy(rew).float().to(device) + gamma*fixed_target_value

                    q_from_net = self.model(torch.from_numpy(curr_state).float().to(device))
                    act_tensor = torch.from_numpy(act).long().to(device)
                    prediction = torch.gather(input=q_from_net, dim=1, index=act_tensor)

                    # loss using MSE
                    loss = loss_fn(target, prediction)
                    episode_loss += loss.item()

                    # backpropagation
                    optimizer.zero_grad()
                    loss.backward()

                    # gradient clipping
                    total_grad_norm += torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    optimizer.step()
                    

                # setting the current observation to the next observation for the next step
                current_state = next_obs

                # updating the fixed targets using polyak update
                with torch.no_grad():
                    for p_target, p in zip(self.fixed_targets.parameters(), self.model.parameters()):
                        p_target.data.mul_(polyak_const)
                        p_target.data.add_((1 - polyak_const) * p.data)
            
            total_reward += episode_reward

            # decaying epsilon
            epsilon -= (0.00015)*epsilon

            # tracking if the goal has been reached
            if has_reached_goal: 
                goal = 1
            else: goal = 0

            # calculating the number of steps taken in the episode
            steps = self.steps - prev_steps

            prev_steps = self.steps

            # plotting using tensorboard
            print(f"Episode {i+1} Reward: {episode_reward} Loss: {episode_loss}")
            
            writer.add_scalar("reward / epsiode", episode_reward, i)
            writer.add_scalar("loss / episode", episode_loss, i)
            writer.add_scalar("exploration rate / episode", epsilon, i)
            writer.add_scalar("Average total grad norm / episode", (total_grad_norm/batch_size), i)
            writer.add_scalar("ending in sucess? / episode", goal, i)
            writer.add_scalar("Steps to reach goal / episode", steps, i)
            writer.flush()

            # saving model
            if (save_path is not None) and ((i+1)%save_freq == 0):
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                try:
                    self.save_model(os.path.join(save_path, "episode"+ str(i+1) + ".pth"))
                except:
                    print("Error in saving model")
    
    def eval(self, num_episodes, path=None):
        if path is not None:
            self.model.load_state_dict(torch.load(path))
        
        self.model.eval()

        total_reward = 0
        successive_runs = 0
        for i in range(num_episodes):
            o = self.env.reset()
            o = self.preprocess_observation(o)
            done = False
            while not done:
                act_continuous, act_discrete = self.get_action(o, 0)
                new_state, reward, done, _ = self.env.step(act_continuous)
                new_state = self.preprocess_observation(new_state)
                total_reward += reward

                self.env.render()

                if done==True and reward == 1:
                    successive_runs += 1

                o = new_state

        print(f"Total episodes run: {num_episodes}")
        print(f"Total successive runs: {successive_runs}")
        print(f"Average reward per episode: {total_reward/num_episodes}")

if __name__ == "__main__":
    env = gym.make("SocNavEnv-v1")
    input_layer_size = env.observation_space["goal"].shape[0] + env.observation_space["humans"].shape[0] + env.observation_space["laptops"].shape[0] + env.observation_space["tables"].shape[0] + env.observation_space["plants"].shape[0]
    model = DQNAgent(input_layer_size, [512, 128, 64, 6], BUFFER_SIZE, env)
    model.train(render=False)
