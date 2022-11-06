import gym
import socnavenv
from socnavenv.wrappers import DiscreteActions
from stable_baselines3 import PPO
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--env_config", help="path to environment config", required=True)
ap.add_argument("-t", "--tensorboard_log", help="name of tensorboard", required=True)
ap.add_argument("-s", "--save_path", help="path to save the model", required=True)
args = vars(ap.parse_args())

env = gym.make("SocNavEnv-v1", config=args["env_config"])
env = DiscreteActions(env)
net_arch = {}
net_arch["pi"] = [512, 256, 128, 64]
net_arch["vf"] = [512, 256, 128, 64]
policy_kwargs = {"net_arch" : [net_arch]}

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=args["tensorboard_log"], policy_kwargs=policy_kwargs)
model.learn(total_timesteps=100000*200)
model.save(args["save_path"])
# model = PPO.load("ppo_sb3")
