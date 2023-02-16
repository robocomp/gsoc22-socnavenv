import gym
import socnavenv
import torch
from socnavenv.wrappers import DiscreteActions
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from agents.models import Transformer
import argparse
from tqdm import tqdm
from stable_baselines3.common.monitor import Monitor


def eval(model, num_episodes, env):
    # intialising metrics
    discomfort_sngnn = 0
    discomfort_crowdnav = 0
    timeout = 0
    success_rate = 0
    time_taken = 0
    closest_human_dist = 0
    closest_obstacle_dist = 0
    collision_rate = 0

    
    total_reward = 0
    print(f"Evaluating model for {num_episodes} episodes")

    for i in tqdm(range(num_episodes)):
        obs, _ = env.reset()
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
            action, _states = model.predict(obs, deterministic=True)
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # env.render()
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
                steps = env.EPISODE_LENGTH
            
            if info["MAX_STEPS"]:
                has_timed_out = 1

            min_human_dist = min(min_human_dist, info["closest_human_dist"])
            min_obstacle_dist = min(min_obstacle_dist, info["closest_obstacle_dist"])

            episode_reward += reward
            
            obs = new_state

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



env = gym.make("SocNavEnv-v1", config="./paper_configs/exp2_no_sngnn.yaml")
env = DiscreteActions(env)

try:
    model = DQN.load("/home/sushant/github/gsoc22-socnavenv/best models/rl_model_9475000_steps.zip")
except Exception as e:
    print(e)
else:
    print("Successfully loaded")

# print(model.policy)

# # eval(model, 100, env)
for _ in range(10):
    done = False
    obs, _ = env.reset()
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        done = terminated or truncated