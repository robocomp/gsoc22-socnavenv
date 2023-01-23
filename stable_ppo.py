import gym
import socnavenv
import torch
from socnavenv.wrappers import DiscreteActions
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from agents.models import Transformer
import argparse
from comet_ml import Experiment
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import ts2xy, plot_results
from stable_baselines3.common.utils import safe_mean

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
    
class CometMLCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, run_name:str, verbose=0):
        super(CometMLCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        print("Logging using comet_ml")
        self.run_name = run_name
        self.experiment = Experiment(
            api_key="8U8V63x4zSaEk4vDrtwppe8Vg",
            project_name="socnav",
            parse_args=False   
        )
        self.experiment.set_name(self.run_name)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        metrics = {
            "rollout/ep_rew_mean": safe_mean([ep_info["r"] for ep_info in self.locals['self'].ep_info_buffer]),
            "rollout/ep_len_mean": safe_mean([ep_info["l"] for ep_info in self.locals['self'].ep_info_buffer])
        }

        l = [
            "train/entropy_loss",
            "train/policy_gradient_loss",
            "train/value_loss",
            "train/approx_kl",
            "train/clip_fraction",
            "train/loss",
            "train/explained_variance"
        ]

        for val in l:
            if val in self.logger.name_to_value.keys():
                metrics[val] = self.logger.name_to_value[val]

        step = self.locals['self'].num_timesteps

        self.experiment.log_metrics(metrics, step=step)
        
    
    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--env_config", help="path to environment config", required=True)
ap.add_argument("-r", "--run_name", help="name of comet_ml run", required=True)
ap.add_argument("-s", "--save_path", help="path to save the model", required=True)
ap.add_argument("-u", "--use_transformer", help="True or False, based on whether you want a transformer based feature extractor", required=False, default=False)
ap.add_argument("-d", "--use_deep_net", help="True or False, based on whether you want a transformer based feature extractor", required=False, default=False)
ap.add_argument("-g", "--gpu", help="gpu id to use", required=False, default="0")
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

device = 'cuda:'+str(args["gpu"]) if torch.cuda.is_available() else 'cpu'
model = PPO("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs, device=device)
callback = CometMLCallback(args["run_name"])
model.learn(total_timesteps=100000*200, callback=callback)
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
