import gym
import socnavenv
import torch
from socnavenv.wrappers import DiscreteActions
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from agents.models import Transformer
import argparse


class TransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 256):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        self.transformer = Transformer(8, 14, 512, 512, None)

        # Update the features dim manually
        self._features_dim = 512

        print("Using transformer for feature extraction")

    def preprocess_observation(self, obs):
        """
        To convert dict observation to numpy observation
        """
        assert(type(obs) == dict)
        observation = torch.tensor([]).float()
        if "goal" in obs.keys() : observation = torch.cat((observation, obs["goal"]) , dim=1)
        if "humans" in obs.keys() : observation = torch.cat((observation, obs["humans"]) , dim=1)
        if "laptops" in obs.keys() : observation = torch.cat((observation, obs["laptops"]) , dim=1)
        if "tables" in obs.keys() : observation = torch.cat((observation, obs["tables"]) , dim=1)
        if "plants" in obs.keys() : observation = torch.cat((observation, obs["plants"]) , dim=1)
        if "walls" in obs.keys():
            observation = torch.cat((observation, obs["walls"]), dim=1)
        return observation

    
    def postprocess_observation(self, obs):
        """
        To convert a one-vector observation into two inputs that can be given to the transformer
        """
        if(len(obs.shape) == 1):
            obs = obs.reshape(1, -1)
        
        robot_state = obs[:, 0:8].reshape(obs.shape[0], -1, 8)
        entity_state = obs[:, 8:].reshape(obs.shape[0], -1, 14)
        
        return robot_state, entity_state

    def forward(self, observations):
        pre = self.preprocess_observation(observations)
        r, e = self.postprocess_observation(pre)
        out = self.transformer(r, e)
        out = out.squeeze(1)
        return out

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--env_config", help="path to environment config", required=True)
ap.add_argument("-t", "--tensorboard_log", help="name of tensorboard", required=True)
ap.add_argument("-s", "--save_path", help="path to save the model", required=True)
ap.add_argument("-u", "--use_transformer", help="True or False, based on whether you want a transformer based feature extractor", required=False, default=False)
ap.add_argument("-d", "--use_deep_net", help="True or False, based on whether you want a transformer based feature extractor", required=False, default=False)
args = vars(ap.parse_args())

env = gym.make("SocNavEnv-v1", config=args["env_config"])
env = DiscreteActions(env)
net_arch = {}
if not args["use_deep_net"]:
    net_arch["pi"] = [512, 256, 128, 64]
    net_arch["vf"] = [512, 256, 128, 64]

else:
    net_arch["pi"] = [512, 256, 256, 256, 128, 128, 64]
    net_arch["vf"] = [512, 256, 256, 256, 128, 128, 64]

if args["use_transformer"]:
    policy_kwargs = {"net_arch" : [net_arch], "features_extractor_class": TransformerExtractor}
else:
    policy_kwargs = {"net_arch" : [net_arch]}

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=args["tensorboard_log"], policy_kwargs=policy_kwargs)
model.learn(total_timesteps=100000*200)
model.save(args["save_path"])

# for inference:
# model = PPO.load("best models/ppo_sb3.zip")

# for i in range(10):
#     obs, _ = env.reset()
#     done  = False
#     while not done:
#         action, _states = model.predict(obs)
#         obs, rewards, terminated, truncated, info = env.step(action)
#         done = terminated or truncated
#         env.render()
