import time

from simplejson import load
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

import socnavenv
from socnavenv import SocNavEnv

env = SocNavEnv()

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

class MyCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(MyCallback, self).__init__(verbose)
        self.env = env
        self.cum_reward = 0
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
        self.cum_reward = 0
    def _on_step(self) -> bool:
        self.cum_reward += self.env.last_reward
    def _on_rollout_end(self) -> None:
        print(f'Cumulative reward: {self.cum_reward}')
    def _on_training_end(self) -> None:
        pass


# action_noise = NormalActionNoise(mean=np.array([0.0, 0.0]), sigma=np.array([1., 2.]))
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.array([ 0.1, 0.0]),
                                            sigma=np.array([0.125, 0.125]),
                                            theta=0.25,
                                            dt=0.5
                                            )

callback = MyCallback(env)

model = DDPG("MlpPolicy",
             env,
             action_noise=action_noise,
             verbose=1,
             tensorboard_log='./logs/', 
             )


try:
    model = DDPG.load("ddpg_socnavenv")
    loaded_from_file = True
except:
    loaded_from_file = False

if not loaded_from_file:
    # Learn and save
    model.learn(total_timesteps=1000000000) # , callback=callback
    model.save("ddpg_socnavenv")
    for i in range(20):
        print('DOOOOOOOOONEEEEEEEE')

# Test the results
obs = env.reset()
for i in range(100000000000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(socnavenv.TIMESTEP)
    if done:
      obs = env.reset()

env.close()
