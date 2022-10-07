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
import yaml
import argparse
from torch.utils.tensorboard import SummaryWriter
from agents.models import MLP, ExperienceReplay

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
        q = v + a - torch.mean(a, dim=1, keepdim=True)
        return q

class DuelingDQNAgent:
    def __init__(self, env:gym.Env, config:str, **kwargs) -> None:
        assert(env is not None and config is not None)
        # initializing the env
        self.env = env
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # agent variables
        self.input_layer_size = None
        self.hidden_layers = None
        self.v_net_layers = None
        self.a_net_layers = None
        self.buffer_size = None
        self.num_episodes = None
        self.epsilon = None
        self.epsilon_decay_rate = None
        self.batch_size = None
        self.gamma = None
        self.lr = None
        self.polyak_const = None
        self.render = None
        self.min_epsilon = None
        self.save_path = None
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

        # declaring the network
        self.duelingDQN = DuelingDQN(self.input_layer_size, self.hidden_layers, self.v_net_layers, self.a_net_layers).to(self.device)
        
        #initializing the fixed targets
        self.fixed_targets = DuelingDQN(self.input_layer_size, self.hidden_layers, self.v_net_layers, self.a_net_layers).to(self.device)
        self.fixed_targets.load_state_dict(self.duelingDQN.state_dict())

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

        if self.input_layer_size is None:
            self.input_layer_size = config["input_layer_size"]
            assert(self.input_layer_size is not None), f"Argument input_layer_size cannot be None"

        if self.hidden_layers is None:
            self.hidden_layers = config["hidden_layers"]
            assert(self.hidden_layers is not None), f"Argument hidden_layers cannot be None"

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

        if self.epsilon is None:
            self.epsilon = config["epsilon"]
            assert(self.epsilon is not None), f"Argument epsilon cannot be None"

        if self.epsilon_decay_rate is None:
            self.epsilon_decay_rate = config["epsilon_decay_rate"]
            assert(self.epsilon_decay_rate is not None), f"Argument epsilon_decay_rate cannot be None"

        if self.batch_size is None:
            self.batch_size = config["batch_size"]
            assert(self.batch_size is not None), f"Argument batch_size cannot be None"

        if self.gamma is None:
            self.gamma = config["gamma"]
            assert(self.gamma is not None), f"Argument gamma cannot be None"

        if self.lr is None:
            self.lr = config["lr"]
            assert(self.lr is not None), f"Argument lr cannot be None"

        if self.polyak_const is None:
            self.polyak_const = config["polyak_const"]
            assert(self.polyak_const is not None), f"Argument polyak_const cannot be None"

        if self.render is None:
            self.render = config["render"]
            assert(self.render is not None), f"Argument render cannot be None"

        if self.min_epsilon is None:
            self.min_epsilon = config["min_epsilon"]
            assert(self.min_epsilon is not None), f"Argument min_epsilon cannot be None"

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
        return observation
    

    def get_action(self, current_state, epsilon):

        if np.random.random() > epsilon:
            # exploit
            with torch.no_grad():
                q = self.duelingDQN(torch.from_numpy(current_state).reshape(1, -1).float().to(self.device))
                action_discrete = torch.argmax(q).item()
                action_continuous = self.env.discrete_to_continuous_action(action_discrete)
                return action_continuous, action_discrete
        
        else:
            # explore
            act = np.random.randint(0, 7)
            return self.env.discrete_to_continuous_action(act), act 
    
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

    def update(self):
        curr_state, rew, act, next_state, d = self.experience_replay.sample_batch(self.batch_size)
                    
        # a_max represents the best action on the next state according to the original network (the network other than the target network)
        a_max = torch.argmax(self.duelingDQN(torch.from_numpy(next_state).float().to(self.device)), keepdim=True, dim=1)
        
        # calculating target value given by r + (gamma * Q(s', a_max, theta')) where theta' is the target network parameters
        # if the transition has done=True, then the target is just r

        # the following calculates Q(s', a) for all a
        q_from_target_net = self.fixed_targets(torch.from_numpy(next_state).float().to(self.device))

        # calculating Q(s', a_max) where a_max was the best action calculated by the original network 
        q_s_prime_a_max = torch.gather(input=q_from_target_net, dim=1, index=a_max)

        # calculating the target. The above quantity is being multiplied element-wise with ~d, so that only the episodes that do not terminate contribute to the second quantity in the additon
        target = torch.from_numpy(rew).float().to(self.device) + self.gamma * (q_s_prime_a_max * (~torch.from_numpy(d).bool().to(self.device)))

        # the prediction is given by Q(s, a). calculting Q(s,a) for all a
        q_from_net = self.duelingDQN(torch.from_numpy(curr_state).float().to(self.device))

        # converting the action array to a torch tensor
        act_tensor = torch.from_numpy(act).long().to(self.device)

        # calculating the prediction as Q(s, a) using the Q from q_from_net and the action from act_tensor
        prediction = torch.gather(input=q_from_net, dim=1, index=act_tensor)

        # loss using MSE
        loss = self.loss_fn(prediction, target)
        self.episode_loss += loss.item()

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        self.total_grad_norm += torch.nn.utils.clip_grad_norm_(self.duelingDQN.parameters(), max_norm=0.5).cpu()
        self.optimizer.step()

    def plot(self, episode):
        self.rewards.append(self.episode_reward)
        self.losses.append(self.episode_loss)
        self.exploration_rates.append(self.epsilon)
        self.grad_norms.append(self.total_grad_norm/self.batch_size)
        self.successes.append(self.has_reached_goal)
        self.collisions.append(self.has_collided)
        self.steps_to_reach.append(self.steps)

        if not os.path.isdir(os.path.join(self.save_path, "plots")):
            os.makedirs(os.path.join(self.save_path, "plots"))

        np.save(os.path.join(self.save_path, "plots", "rewards"), np.array(self.rewards), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "losses"), np.array(self.episode_loss), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "exploration_rates"), np.array(self.epsilon), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "grad_norms"), np.array(self.total_grad_norm/self.batch_size), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "successes"), np.array(self.has_reached_goal), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "collisions"), np.array(self.has_collided), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "steps_to_reach"), np.array(self.steps), allow_pickle=True, fix_imports=True)

        self.writer.add_scalar("reward / epsiode", self.episode_reward, episode)
        self.writer.add_scalar("loss / episode", self.episode_loss, episode)
        self.writer.add_scalar("exploration rate / episode", self.epsilon, episode)
        self.writer.add_scalar("Average total grad norm / episode", (self.total_grad_norm/self.batch_size), episode)
        self.writer.add_scalar("ending in sucess? / episode", self.has_reached_goal, episode)
        self.writer.add_scalar("has collided? / episode", self.has_collided, episode)
        self.writer.add_scalar("Steps to reach goal / episode", self.steps, episode)
        self.writer.flush()  

    def train(self):
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.duelingDQN.parameters(), lr=self.lr)
        self.rewards = []
        self.losses = []
        self.exploration_rates = []
        self.grad_norms = []
        self.successes = []
        self.collisions = []
        self.steps_to_reach = []

        self.average_reward = 0

        # train loop
        for i in range(self.num_episodes):
            current_obs, _ = self.env.reset()
            current_obs = self.preprocess_observation(current_obs)
            done = False
            self.episode_reward = 0
            self.total_grad_norm = 0
            self.episode_loss = 0
            self.has_reached_goal = 0
            self.has_collided = 0
            self.steps = 0

            
            while not done: 
                # sampling an action from the current state
                action_continuous, action_discrete = self.get_action(current_obs, self.epsilon)

                # taking a step in the environment
                next_obs, reward, terminated, truncated, info = self.env.step(action_continuous)
                done = terminated or truncated

                # incrementing total steps
                self.steps += 1

                # preprocessing the observation, i.e padding the observation with zeros if it is lesser than the maximum size
                next_obs = self.preprocess_observation(next_obs)
                
                # rendering if reqd
                if self.render and ((i+1) % self.render_freq == 0):
                    self.env.render()

                # storing the rewards
                self.episode_reward += reward

                # storing whether the agent reached the goal
                if info["REACHED_GOAL"]:
                    self.has_reached_goal = 1
                
                if info["COLLISION"]:
                    self.has_collided = 1
                    self.steps = self.env.EPISODE_LENGTH

                # storing the current state transition in the replay buffer. 
                self.experience_replay.insert((current_obs, reward, action_discrete, next_obs, done))


                # sampling a mini-batch of state transitions if the replay buffer has sufficent examples
                if len(self.experience_replay) > self.batch_size:
                    self.update()

                # setting the current observation to the next observation
                current_obs = next_obs

                # updating the fixed targets using polyak update
                with torch.no_grad():
                    for p_target, p in zip(self.fixed_targets.parameters(), self.duelingDQN.parameters()):
                        p_target.data.mul_(self.polyak_const)
                        p_target.data.add_((1 - self.polyak_const) * p.data)


            # decaying epsilon
            if self.epsilon > self.min_epsilon:
                self.epsilon -= (self.epsilon_decay_rate)*self.epsilon

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
            self.duelingDQN.load_state_dict(torch.load(path, map_location=torch.device(self.device)))
        
        self.duelingDQN.eval()

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
                act_continuous, act_discrete = self.get_action(o, 0)
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
