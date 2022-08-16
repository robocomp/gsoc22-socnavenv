import time

from simplejson import load
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

import socnavenv
from socnavenv import SocNavEnv
import mynoise

from stable_baselines3.common.env_checker import check_env

env = SocNavEnv()
from mycallback import MyCallback


# action_noise = NormalActionNoise(mean=np.array([0.0, 0.0]), sigma=np.array([1., 2.]))
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.array([ 0.1, 0.0]),
#                                             sigma=np.array([0.125, 0.125]),
#                                             # theta=0.25,
#                                             # dt=0.5
#                                             )

action_noise = mynoise.MyActionNoise(
    mean=np.array([0.0, 0.0]),
    sigma=np.array([0.0, 0.5])
    )

callback = MyCallback(env)


model = PPO("MlpPolicy",
             env,
             #action_noise=action_noise,
             verbose=1,
             tensorboard_log='./logs/', 
             )


try:
    model = PPO.load("ppo_socnavenv")
    loaded_from_file = True
except:
    loaded_from_file = False

if not loaded_from_file:
    # Learn and save
    model.learn(total_timesteps=1000000000) # , callback=callback
    model.save("ppo_socnavenv")
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
