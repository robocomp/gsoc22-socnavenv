# SocNavEnv : An environment for Social Navigation

## Dependencies
The following dependencies can be installed using pip or Anaconda: `gym` `matplotlib` `opencv-python`.

RVO2 can be installed using the following repository: https://github.com/sybrenstuvel/Python-RVO2/

## Usage
```python
import socnavenv
import gym
env = gym.make('SocNavEnv-v1')
```
## Sample Code
```python
import socnavenv
import gym
env = gym.make("SocNavEnv-v1") 
obs = env.reset()


for i in range(1000):
    obs, reward, done, _ = env.step(env.action_space.sample())
    env.render()
    if done == True:
        env.reset()
```
