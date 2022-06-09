import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import socnavenv
import gym
import numpy as np
import copy
import random
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

###### HYPERPARAMETERS############
LR = 0.0001
BUFFER_SIZE = 2000
BATCH_SIZE = 64
GAMMA = 0.99
NUM_EPISODES = 20000
EPSILON = 0.2
UPDATE_FREQ = 20
##################################

class MLP(nn.Module):
    def __init__(self, input_layer_size:int, hidden_layers:list, last_relu=False) -> None:
        super().__init__()
        self.layers = []
        self.layers.append(nn.Linear(input_layer_size, hidden_layers[0]))
        self.layers.append(nn.ReLU())
        for i in range(len(hidden_layers)-1):
            if i != (len(hidden_layers)-2):
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                self.layers.append(nn.ReLU())
            elif last_relu:
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                self.layers.append(nn.ReLU())
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

    def sample_batch(self, batch_size:int, observation_size:int, action_size:int):
        data = np.array(random.sample(self.list, batch_size))
        curr_state = data[:,0:observation_size]
        reward = data[:,observation_size].reshape(-1, 1)
        action = data[:,observation_size+1].reshape(-1,1)
        next_state = data[:,observation_size+2: 2*observation_size+2]
        done = data[:,-1].reshape(-1,1)

        return (curr_state, reward, action, next_state, done)

class DQN(nn.Module):
    def __init__(self, input_layer_size, hidden_layers):
        super(DQN, self).__init__() 
        self.linear_stack = MLP(input_layer_size, hidden_layers)
        
    def forward(self, x):
        x = self.linear_stack(x)
        return x    

class DQNAgent:
    def __init__(self, input_layer_size, hidden_layers, max_capacity) -> None:
        # initializing the env
        self.env = gym.make("SocNavEnv-v1")
        
        # declaring the network
        self.model = DQN(input_layer_size, hidden_layers).to(device)
        
        #initializing the fixed targets
        self.fixed_targets = DQN(input_layer_size, hidden_layers).to(device)

        # initalizing the replay buffer
        self.experience_replay = ExperienceReplay(max_capacity)

        # variable to keeo count of the number of steps that has occured
        self.steps = 0

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()
        self.gamma = GAMMA
        self.total_reward = 0

    def preprocess_observation(self, obs, MAX_OBSERVATION_LENGTH):
        """
        To pad zeros to the observation if its length is less than the maximum observation size.
        """
        zer = np.zeros(MAX_OBSERVATION_LENGTH-obs.shape[0])
        obs = np.concatenate((obs,zer))
        return obs
    
    
    def discrete_to_continuous_action(self, action:int):
        """
        Function to return a continuous space action for a given discrete action
        """
        if action == 0:
            # move forward with full speed
            return np.array([1, 0], dtype=np.float32) # gives only linear velocity
        
        elif action == 1:
            # move forward with half speed
            return np.array([0, 0], dtype=np.float32) # gives only linear velocity

        elif action == 2:
            # turn leftwards
            return np.array([-1, np.pi/6], dtype=np.float32) # gives only angular velocity in the counter-clockwise direction
        
        elif action == 3:
            # turn rightwards
            return np.array([-1, -np.pi/6], dtype=np.float32) # gives only angular velocity in the counter-clockwise direction
        
        else:
            raise NotImplementedError

    def get_action(self, current_state, epsilon):
        num = np.random.random()
        
        if num > epsilon:
            # exploit
            q = self.model(torch.from_numpy(current_state).reshape(1, -1).float().to(device))
            action_discrete = torch.argmax(q).item()
            action_continuous = self.discrete_to_continuous_action(action_discrete)
            return action_continuous, action_discrete
        
        else:
            # explore
            act = np.random.randint(0, 4)
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
        update_freq=UPDATE_FREQ,
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
            current_state = self.preprocess_observation(current_state, self.env.MAX_OBSERVATION_LENGTH)
            
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
                next_obs = self.preprocess_observation(next_obs, self.env.MAX_OBSERVATION_LENGTH)
                
                # rendering if reqd
                if render and ((i+1) % render_freq == 0):
                    self.env.render()

                # storing the rewards
                episode_reward += reward

                # storing whether the agent reached the goal
                if reward == 1 and done == True:
                    has_reached_goal = True

                # storing the current state transition in the replay buffer. 
                arr = current_state.copy()
                arr = np.concatenate((arr, np.array([reward], dtype=np.float32)))
                arr = np.concatenate((arr, np.array([action_discrete], dtype=np.float32)))
                arr = np.concatenate((arr, next_obs))
                arr = np.concatenate((arr, np.array([done], dtype=np.float32)))
                self.experience_replay.insert(arr)
                
                if len(self.experience_replay) > batch_size:
                    curr_state, rew, act, next_state, d = self.experience_replay.sample_batch(batch_size, self.env.MAX_OBSERVATION_LENGTH, 1)
                    # fixed_target_value = torch.max(self.fixed_targets(torch.from_numpy(next_state).float().to(device)), dim=1, keepdim=True).values
                    fixed_target_value = self.fixed_targets(torch.from_numpy(next_state).float().to(device))
                    fixed_target_value = torch.max(fixed_target_value, dim=1, keepdim=True).values
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
                    optimizer.step()

                # setting the current observation to the next observation for the next step
                current_state = next_obs

                # updating the fixed targets
                if self.steps % update_freq == 0:
                    self.fixed_targets.load_state_dict(self.model.state_dict())
                
                total_grad_norm += self.calculate_grad_norm()

            
            total_reward += episode_reward

            # decaying epsilon
            epsilon -= (0.0005)*epsilon

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
            writer.add_scalar("total grad norm / episode", total_grad_norm, i)
            writer.add_scalar("ending in sucess? / episode", goal, i)
            writer.add_scalar("Steps to reach goal / episode", steps, i)
            writer.flush()

            # saving model
            if (save_path is not None) and ((i+1)%save_freq == 0):
                try:
                    self.save_model(save_path + "_episode"+ str(i+1) + ".pth")
                except:
                    print(f"Path {save_path} does not exist!")

if __name__ == "__main__":
    model = DQNAgent(242, [400, 128, 32, 4], BUFFER_SIZE)
    model.train(render=True)
