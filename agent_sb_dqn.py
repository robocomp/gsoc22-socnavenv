import time
import torch
from simplejson import load
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
import socnavenv
from socnavenv import DiscreteSocNavEnv


from stable_baselines3.common.callbacks import BaseCallback
class MyCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(MyCallback, self).__init__(verbose)
        # self.model
        # self.model.get_env()
        # self.training_env
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # self.logger = None  # type: logger.Logger
        # self.parent = None  # type: Optional[BaseCallback]
    def _on_training_start(self) -> None:
        pass
    def _on_rollout_start(self) -> None:
        pass
    def _on_step(self) -> bool:
        pass
    def _on_rollout_end(self) -> None:
        pass
    def _on_training_end(self) -> None:
        pass


env = DiscreteSocNavEnv(advance_split=3, rotation_split=3)


policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[400,400,400,300])


model = DQN("MlpPolicy",
             env,
             verbose=1,
             tensorboard_log='./logs/', 
             learning_rate=1e-3,
             batch_size=128,
             tau=0.05,
             train_freq=3,
             target_update_interval=1000,
             buffer_size=10_000_000,
             exploration_fraction=0.1,
             policy_kwargs=policy_kwargs
             )

try:
    model = DQN.load("dqn_socnavenv")
    loaded_from_file = True
except:
    loaded_from_file = False

if not loaded_from_file:
    # Learn and save
    model.learn(total_timesteps=10_000_000) # , callback=MyCallback(env))
    model.save("dqn_socnavenv")
    print('DONE')

socnavenv.DEBUG = 2
socnavenv.MILLISECONDS = 1

# Test the results
obs = env.reset()
for i in range(100000000000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
