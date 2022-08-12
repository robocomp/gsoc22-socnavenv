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
import os
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# hparams
LR = 0.001
BUFFER_SIZE = 200000
BATCH_SIZE = 32
GAMMA = 0.99
NUM_EPISODES = 100000
EPSILON = 1
POLYAK_CONSTANT = 0.995
MIN_EPSILON=0.15

min_epsilon=MIN_EPSILON,

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

class Embedding(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.linear(x)
        return x

class Transformer(nn.Module):
    def __init__(self, input_emb1:int, input_emb2:int, d_model:int, d_k:int, mlp_hidden_layers:list) -> None:
        super().__init__()
        self.embedding1 = Embedding(input_dim=input_emb1, output_dim=d_model)
        self.embedding2 = Embedding(input_dim=input_emb2, output_dim=d_model)
        self.key_net = nn.Linear(d_model, d_k)
        self.query_net = nn.Linear(d_model, d_k)

        self.softmax = nn.Softmax(dim=2)
        self.mlp = MLP(2*d_model, mlp_hidden_layers)

    def forward(self, inp1, inp2):
        embedding1 = self.embedding1(inp1)
        embedding2 = self.embedding2(inp2)
        q = self.query_net(embedding1)
        k = self.key_net(embedding2)
        attention_matrix = self.softmax(torch.matmul(q, k.transpose(1,2)))
        attention_value = torch.matmul(attention_matrix, embedding2)
        x = torch.cat((embedding1, attention_value), dim=2)
        q = self.mlp(x)
        return q

class ExperienceReplay:
    def __init__(self, max_capacity) -> None:
        self.list = deque(maxlen = max_capacity)

    def insert(self, val:tuple) -> None:
        # (current_state, reward, action, next_state, done)
        self.list.append(val)

    def __len__(self):
        return len(self.list)

    def sample_batch(self, batch_size:int):
        sample = random.sample(self.list, batch_size)
        current_state, reward, action, next_state, done = zip(*sample)

        current_state = list(current_state)
        maxi = -1
        for arr in current_state:
            maxi = max(maxi, arr.shape[0])
        for i in range(len(current_state)):
            current_state[i] = np.concatenate((current_state[i], np.zeros(maxi-current_state[i].shape[0])))

        next_state = list(next_state)
        maxi = -1
        for arr in next_state:
            maxi = max(maxi, arr.shape[0])
        for i in range(len(next_state)):
            next_state[i] = np.concatenate((next_state[i], np.zeros(maxi-next_state[i].shape[0])))
        
        current_state = np.array(current_state)
        reward = np.array(reward).reshape(-1, 1)
        action = np.array(action).reshape(-1, 1)
        next_state = np.array(next_state)
        done = np.array(done).reshape(-1, 1)
        
        return current_state, reward, action, next_state, done 


class DuelingDQN_Transformer(nn.Module):
    def __init__(self, input_emb1:int, input_emb2:int, d_model:int, d_k:int, mlp_hidden_layers:list, v_net_layers:list, a_net_layers:list) -> None:
        super().__init__()
        # sizes of the first layer in the value and advantage networks should be same as the output of the hidden layer network
        assert(v_net_layers[0]==mlp_hidden_layers[-1] and a_net_layers[0]==mlp_hidden_layers[-1])
        
        self.transformer = Transformer(input_emb1, input_emb2, d_model, d_k, mlp_hidden_layers)
        self.value_network = MLP(v_net_layers[0], v_net_layers[1:])
        self.advantage_network = MLP(a_net_layers[0], a_net_layers[1:])


    def forward(self, inp1, inp2):
        h = self.transformer.forward(inp1, inp2)
        v = self.value_network.forward(h)
        a = self.advantage_network.forward(h)
        q = v + a - torch.mean(a, dim=2, keepdim=True)
        return q

class DuelingDQN_Transformer_Agent:
    def __init__(self, input_emb1:int, input_emb2:int, d_model:int, d_k:int, mlp_hidden_layers:list, v_net_layers:list, a_net_layers:list, max_capacity:int, env, run_name=None) -> None:
        # initializing the env
        self.env = env

        # declaring the network
        self.duelingDQN = DuelingDQN_Transformer(input_emb1, input_emb2, d_model, d_k, mlp_hidden_layers, v_net_layers, a_net_layers).to(device)
        
        # initializing using xavier initialization
        self.duelingDQN.apply(self.xavier_init_weights)

        #initializing the fixed targets
        self.fixed_targets = DuelingDQN_Transformer(input_emb1, input_emb2, d_model, d_k, mlp_hidden_layers, v_net_layers, a_net_layers).to(device)
        self.fixed_targets.load_state_dict(self.duelingDQN.state_dict())

        # initalizing the replay buffer
        self.experience_replay = ExperienceReplay(max_capacity)

        # variable to keep count of the number of steps that has occured
        self.steps = 0

        if run_name is not None:
            self.writer = SummaryWriter('runs/'+run_name)
        else:
            self.writer = SummaryWriter()

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

    def get_action(self, current_state, epsilon):

        if np.random.random() > epsilon:
            # exploit
            with torch.no_grad():
                robot_state, entity_state = self.postprocess_observation(current_state)
                q = self.duelingDQN(torch.from_numpy(robot_state).float().to(device), torch.from_numpy(entity_state).float().to(device))
                action_discrete = torch.argmax(q.squeeze(0)).item()
                action_continuous = self.discrete_to_continuous_action(action_discrete)
                return action_continuous, action_discrete
        
        else:
            # explore
            act = np.random.randint(0, 6)
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
        epsilon_decay_rate=0,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        lr = LR,
        polyak_const=POLYAK_CONSTANT,
        render=False,
        min_epsilon=MIN_EPSILON,
        save_path = 2,
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
            current_obs = self.preprocess_observation(current_obs)
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
                self.experience_replay.insert((current_obs, reward, action_discrete, next_obs, done))


                # sampling a mini-batch of state transitions if the replay buffer has sufficent examples
                if len(self.experience_replay) > batch_size:
                    curr_state, rew, act, next_state, d = self.experience_replay.sample_batch(batch_size)
                    
                    # getting robot and entity observations from next_state
                    next_state_robot, next_state_entity = self.postprocess_observation(next_state)

                    # a_max represents the best action on the next state according to the original network (the network other than the target network)
                    a_max = torch.argmax(
                        self.duelingDQN(
                            torch.from_numpy(next_state_robot).float().to(device), 
                            torch.from_numpy(next_state_entity).float().to(device)
                        ), 
                        keepdim=True, 
                        dim=2).squeeze(1)
                    
                    # calculating target value given by r + (gamma * Q(s', a_max, theta')) where theta' is the target network parameters
                    # if the transition has done=True, then the target is just r

                    # the following calculates Q(s', a) for all a
                    q_from_target_net = self.fixed_targets(
                        torch.from_numpy(next_state_robot).float().to(device), 
                        torch.from_numpy(next_state_entity).float().to(device)
                    ).squeeze(1)

                    # calculating Q(s', a_max) where a_max was the best action calculated by the original network 
                    q_s_prime_a_max = torch.gather(input=q_from_target_net, dim=1, index=a_max)

                    # calculating the target. The above quantity is being multiplied element-wise with ~d, so that only the episodes that do not terminate contribute to the second quantity in the additon
                    target = torch.from_numpy(rew).float().to(device) + gamma * (q_s_prime_a_max * (~torch.from_numpy(d).bool().to(device)))

                    # getting robot and entity observations from curr_state
                    curr_state_robot, curr_state_entity = self.postprocess_observation(curr_state)

                    # the prediction is given by Q(s, a). calculting Q(s,a) for all a
                    q_from_net = self.duelingDQN(
                        torch.from_numpy(curr_state_robot).float().to(device), 
                        torch.from_numpy(curr_state_entity).float().to(device)
                    ).squeeze(1)

                    # converting the action array to a torch tensor
                    act_tensor = torch.from_numpy(act).long().to(device)

                    # calculating the prediction as Q(s, a) using the Q from q_from_net and the action from act_tensor
                    prediction = torch.gather(input=q_from_net, dim=1, index=act_tensor)

                    # loss using MSE
                    loss = loss_fn(prediction, target)
                    episode_loss += loss.item()

                    # backpropagation
                    optimizer.zero_grad()
                    loss.backward()

                    # gradient clipping
                    total_grad_norm += torch.nn.utils.clip_grad_norm_(self.duelingDQN.parameters(), max_norm=0.5)
                    optimizer.step()

                # setting the current observation to the next observation
                current_obs = next_obs

                # updating the fixed targets using polyak update
                with torch.no_grad():
                    for p_target, p in zip(self.fixed_targets.parameters(), self.duelingDQN.parameters()):
                        p_target.data.mul_(polyak_const)
                        p_target.data.add_((1 - polyak_const) * p.data)

            total_reward += episode_reward

            # decaying epsilon
            if epsilon > min_epsilon:
                epsilon -= (epsilon_decay_rate)*epsilon

            if has_reached_goal: 
                goal = 1
            else: goal = 0
            
            steps = self.steps - prev_steps

            prev_steps = self.steps  

            # plotting using tensorboard
            print(f"Episode {i+1} Reward: {episode_reward} Loss: {episode_loss}")
            
            self.writer.add_scalar("reward / epsiode", episode_reward, i)
            self.writer.add_scalar("loss / episode", episode_loss, i)
            self.writer.add_scalar("exploration rate / episode", epsilon, i)
            self.writer.add_scalar("Average total grad norm / episode", (total_grad_norm/batch_size), i)
            self.writer.add_scalar("ending in sucess? / episode", goal, i)
            self.writer.add_scalar("Steps to reach goal / episode", steps, i)
            self.writer.flush()

            # saving model
            if (save_path is not None) and ((i+1)%save_freq == 0):
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                try:
                    self.save_model(os.path.join(save_path, "episode"+ str(i+1).zfill(8) + ".pth"))
                except:
                    print("Error in saving model")
   
    def eval(self, num_episodes, path=None):
        if path is not None:
            self.duelingDQN.load_state_dict(torch.load(path, map_location=torch.device(device)))
        
        self.duelingDQN.eval()

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

    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--env_config", required=True, help="path to env config")
    ap.add_argument("-c", "--config", required=True, help="path to config file")
    ap.add_argument("-r", "--run_name", required=False, default=None)
    args = vars(ap.parse_args())

    config = args["config"]
    env.configure(args["env_config"])

    # reading config file
    with open(config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    robot_state_dim = env.observation_space["goal"].shape[0]
    entity_state_dim = 13
    model = DuelingDQN_Transformer_Agent(robot_state_dim, entity_state_dim, 6, 5, config["hidden_layers"], config["v_net_layers"], config["a_net_layers"], config["buffer_size"], env, args["run_name"])
    model.train(
        num_episodes=config["num_episodes"],
        epsilon=config["epsilon"],
        epsilon_decay_rate=config["epsilon_decay_rate"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        lr=config["lr"],
        polyak_const=config["polyak_constant"],
        render=config["render"],
        min_epsilon=config["min_epsilon"] ,
        save_path=config["save_path"],
        render_freq=config["render_freq"],
        save_freq=config["save_freq"]
    )
    