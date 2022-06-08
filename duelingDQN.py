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

# hparams
BUFFER_SIZE = 10000
UPDATE_FREQ = 20
NUM_EPISODES = 10000
EPSILON = 0.2
BATCH_SIZE = 256
GAMMA = 0.9
LR = 0.001

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

    def insert(self, val:np.ndarray) -> None:
        # (current_state, reward, action, next_state, done)
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



class DuelingDQN(nn.Module):
    def __init__(self, input_size, hidden_layers:list, v_net_layers:list, a_net_layers:list) -> None:
        super().__init__()
        # sizes of the first layer in the value and advantage networks should be same as the output of the hidden layer network
        assert(v_net_layers[0]==hidden_layers[-1] and a_net_layers[0]==hidden_layers[-1])
        self.hidden_mlp = MLP(input_size, hidden_layers)
        self.value_network = MLP(v_net_layers[0], v_net_layers[1:])
        self.advantage_network = MLP(a_net_layers[0], a_net_layers[1:])
        

    def forward(self,x):
        x = self.hidden_mlp.forward(x)
        v = self.value_network.forward(x)
        a = self.advantage_network.forward(x)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q

class DuelingDQNAgent:
    def __init__(self, input_size, hidden_layers:list, v_net_layers:list, a_net_layers:list, max_capacity:int) -> None:
        # initializing the env
        self.env = gym.make("SocNavEnv-v1")

        # declaring the network
        self.duelingDQN = DuelingDQN(input_size, hidden_layers, v_net_layers, a_net_layers).to(device)

        #initializing the fixed targets
        self.fixed_targets = copy.deepcopy(self.duelingDQN).to(device)

        # initalizing the replay buffer
        self.experience_replay = ExperienceReplay(max_capacity)

        # variable to keeo count of the number of steps that has occured
        self.steps = 0

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
            return np.array([-1, np.pi/4], dtype=np.float32) # gives only angular velocity in the counter-clockwise direction
        
        elif action == 3:
            # turn rightwards
            return np.array([-1, -np.pi/4], dtype=np.float32) # gives only angular velocity in the counter-clockwise direction
        
        else:
            raise NotImplementedError

    def get_action(self, current_state, epsilon):
        num = np.random.random()
        
        if num > epsilon:
            # exploit
            q = self.duelingDQN(torch.from_numpy(current_state).reshape(1, -1).float().to(device))
            action_discrete = torch.argmax(q).item()
            action_continuous = self.discrete_to_continuous_action(action_discrete)
            return action_continuous, action_discrete
        
        else:
            # explore
            act = np.random.randint(0, 4)
            return self.discrete_to_continuous_action(act), act 
    
    def calculate_grad_norm(self):
        total_norm = 0
        parameters = [p for p in self.duelingDQN.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def save_model(self, path):
        torch.save(self.duelingDQN.state_dict(), path)

    def train(
        self,
        num_episodes=NUM_EPISODES,
        epsilon=EPSILON,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        lr = LR,
        update_freq=UPDATE_FREQ,
        render=False,
        save_path = "./models/duelingdqn",
        render_freq = 500,
        save_freq = 500
    ):
        total_reward = 0
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.duelingDQN.parameters(), lr=lr)
        prev_steps = 0 # denotes the number of steps already taken by the agent before the start of the episode
        
        # train loop
        for i in range(num_episodes):
            current_obs = self.env.reset()
            current_obs = self.preprocess_observation(current_obs, self.env.MAX_OBSERVATION_LENGTH)
            done = False
            episode_reward = 0
            episode_loss = 0
            total_grad_norm = 0
            has_reached_goal = False
            
            while not done: 
                # sampling an action from the current state
                action_continuous, action_discrete = self.get_action(current_obs, epsilon)

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
                arr = current_obs.copy()
                arr = np.concatenate((arr, np.array([reward], dtype=np.float32)))
                arr = np.concatenate((arr, np.array([action_discrete], dtype=np.float32)))
                arr = np.concatenate((arr, next_obs))
                arr = np.concatenate((arr, np.array([done], dtype=np.float32)))
                self.experience_replay.insert(arr)

                # sampling a mini-batch of state transitions if the replay buffer has sufficent examples
                if len(self.experience_replay) > batch_size:
                    curr_state, rew, act, next_state, d = self.experience_replay.sample_batch(batch_size, self.env.MAX_OBSERVATION_LENGTH, 1)
                    
                    # a_max represents the best action on the next state according to the original network (the network other than the target network)
                    a_max = torch.argmax(self.duelingDQN(torch.from_numpy(next_state).float().to(device)), keepdim=True, dim=1)
                    
                    # calculating target value given by r + (gamma * Q(s', a_max, theta')) where theta' is the target network parameters
                    # if the transition has done=True, then the target is just r

                    # the following calculates Q(s', a) for all a
                    q_from_target_net = self.fixed_targets(torch.from_numpy(next_state).float().to(device))

                    # calculating Q(s', a_max) where a_max was the best action calculated by the original network 
                    q_s_prime_a_max = torch.gather(input=q_from_target_net, dim=1, index=a_max)

                    # calculating the target. The above quantity is being multiplied element-wise with ~d, so that only the episodes that do not terminate contribute to the second quantity in the additon
                    target = torch.from_numpy(rew).float().to(device) + gamma * (q_s_prime_a_max * (~torch.from_numpy(d).bool().to(device)))

                    # the prediction is given by Q(s, a). calculting Q(s,a) for all a
                    q_from_net = self.duelingDQN(torch.from_numpy(curr_state).float().to(device))

                    # converting the action array to a torch tensor
                    act_tensor = torch.from_numpy(act).long().to(device)

                    # calculating the prediction as Q(s, a) using the Q from q_from_net and the action from act_tensor
                    prediction = torch.gather(input=q_from_net, dim=1, index=act_tensor)

                    # loss using MSE
                    loss = loss_fn(target, prediction)
                    episode_loss += loss.item()

                    # backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # setting the current observation to the next observation
                current_obs = next_obs

                if self.steps % update_freq == 0:
                    self.fixed_targets = copy.deepcopy(self.duelingDQN)

                total_grad_norm += self.calculate_grad_norm()

            total_reward += episode_reward

            # decaying epsilon
            epsilon -= (0.0005)*epsilon

            if has_reached_goal: 
                goal = 1
            else: goal = 0
            
            steps = self.steps - prev_steps

            prev_steps = self.steps  

            # plotting using tensorboard
            print(f"Episode {i+1} Reward: {episode_reward} Loss: {episode_loss}")
            
            writer.add_scalar("reward / epsiode", episode_reward, i)
            writer.add_scalar("loss / episode", episode_loss, i)
            writer.add_scalar("exploration rate / episode", epsilon, i)
            writer.add_scalar("total grad norm / episode", epsilon, i)
            writer.add_scalar("ending in sucess? / episode", goal, i)
            writer.add_scalar("Steps to reach goal / episode", steps, i)
            writer.flush()

            # saving model
            if (save_path is not None) and ((i+1)%save_freq == 0):
                try:
                    self.save_model(save_path + "_episode"+ str(i+1) + ".pth")
                except:
                    print(f"Path {save_path} does not exist!")
   
    def eval(self, num_episodes, path=None):
        if path is not None:
            self.duelingDQN.load_state_dict(torch.load(path))
        
        self.duelingDQN.eval()

        total_reward = 0
        successive_runs = 0
        for i in range(num_episodes):
            o = self.env.reset()
            o = self.preprocess_observation(o, self.env.MAX_OBSERVATION_LENGTH)
            done = False
            while not done:
                act_continuous, act_discrete = self.get_action(o, 0)
                new_state, reward, done, _ = self.env.step(act_continuous)
                new_state = self.preprocess_observation(new_state, self.env.MAX_OBSERVATION_LENGTH)
                total_reward += reward

                self.env.render()

                if done==True and reward == 1:
                    successive_runs += 1

                o = new_state

        print(f"Total episodes run: {num_episodes}")
        print(f"Total successive runs: {successive_runs}")
        print(f"Average reward per episode: {total_reward/num_episodes}")


if __name__ == "__main__":
    model = DuelingDQNAgent(242, [120, 60, 40], [40, 20, 5, 1], [40, 20, 10, 4], 2000)
    model.train(render=True)
