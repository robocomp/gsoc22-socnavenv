# SocNavEnv : An environment for Social Navigation

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