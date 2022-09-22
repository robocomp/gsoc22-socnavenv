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
from torch.utils.data import Dataset

class MLP(nn.Module):
    """
    Class for a Multi Layered Perceptron. LeakyReLU activations would be applied between each layer.

    Args:
    input_layer_size (int): The size of the input layer
    hidden_layers (list): A list containing the sizes of the hidden layers
    last_relu (bool): If True, then a LeakyReLU would be applied after the last hidden layer
    """
    def __init__(self, input_layer_size:int, hidden_layers:list, last_relu=False) -> None:
        super().__init__()
        self.layers = []
        self.layers.append(nn.Linear(input_layer_size, hidden_layers[0]))
        self.layers.append(nn.LeakyReLU())
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.layers[-2].weight, gain=gain)
        for i in range(len(hidden_layers)-1):
            if i != (len(hidden_layers)-2):
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                self.layers.append(nn.LeakyReLU())
                nn.init.xavier_uniform_(self.layers[-2].weight, gain=gain)
            elif last_relu:
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                self.layers.append(nn.LeakyReLU())
                nn.init.xavier_uniform_(self.layers[-2].weight, gain=gain)
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
        assert(len(val) == 5)
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

class Embedding(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(),
            )
        self.set_parameters()

    def set_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        # gain_last_layer = nn.init.calculate_gain('tanh', 0.01)
        nn.init.xavier_uniform_(self.linear[0].weight, gain=gain)
    
    def forward(self, x):
        x = self.linear(x)
        return x

class Transformer(nn.Module):
    def __init__(self, input_emb1:int, input_emb2:int, d_model:int, d_k:int, mlp_hidden_layers:list) -> None:
        super().__init__()
        self.embedding1 = Embedding(input_dim=input_emb1, output_dim=d_model)
        self.embedding2 = Embedding(input_dim=input_emb2, output_dim=d_model)
        self.key_net = nn.Sequential(
            nn.Linear(d_model, d_k),
            nn.LeakyReLU()
            )
        self.query_net = nn.Sequential(
            nn.Linear(d_model, d_k),
            nn.LeakyReLU()
            )

        self.attention_net = nn.Sequential(
            nn.Linear(d_model, d_k),
            nn.LeakyReLU()
        )
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = MLP(2*d_model, mlp_hidden_layers) if mlp_hidden_layers is not None else None

        self.set_parameters()

    def set_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        # gain_last_layer = nn.init.calculate_gain('leaky_relu', 0.01)
        nn.init.xavier_uniform_(self.key_net[0].weight, gain=gain)
        nn.init.xavier_uniform_(self.query_net[0].weight, gain=gain)
        nn.init.xavier_uniform_(self.attention_net[0].weight, gain=gain)


    def forward(self, inp1, inp2):
        embedding1 = self.embedding1(inp1)
        embedding2 = self.embedding2(inp2)
        q = self.query_net(embedding1)
        k = self.key_net(embedding2)
        a = self.attention_net(embedding2)
        attention_matrix = self.softmax(torch.matmul(q, k.transpose(1,2)))
        attention_value = torch.matmul(attention_matrix, a)
        x = torch.cat((embedding1, attention_value), dim=-1)
        if self.mlp is not None:
            q = self.mlp(x)
            return q
        else:
            return x

class RolloutBuffer:
	def __init__(self):
		self.states = []
		self.probs = []
		self.logprobs = []
		self.actions = []
		self.one_hot_actions = []
		self.rewards = []
		self.dones = []
		self.values = []
		self.qvalues = []
	

	def clear(self):
		del self.actions[:]
		del self.states[:]
		del self.probs[:]
		del self.one_hot_actions[:]
		del self.logprobs[:]
		del self.rewards[:]
		del self.dones[:]
		del self.values[:]
		del self.qvalues[:]

class CrowdNavMemory:
    def __init__(self, max_capacity) -> None:
        self.list = deque(maxlen = max_capacity)
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'

    def push(self, val) -> None:
        assert(len(val) == 2)
        self.list.append(val)

    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, i):
        return torch.stack([self.list[i][0]]), torch.stack([self.list[i][1]]).to(self.device)

    def sample_batch(self, batch_size:int):
        sample = random.sample(self.list, batch_size)
        current_state, values = zip(*sample)
        current_state = list(current_state)
        maxi = -1
        for arr in current_state:
            maxi = max(maxi, arr.shape[0])
        
        for i in range(len(current_state)):
            current_state[i] = torch.cat((current_state[i], torch.zeros(maxi-current_state[i].shape[0], current_state[i].shape[1]).to(self.device)), 0)

        current_state = torch.stack(current_state).to(self.device)
        values = torch.stack(values).to(self.device)
        return current_state, values