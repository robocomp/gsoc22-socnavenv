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


goal = [obs[0], obs[1]]
object_obs = obs[2:].reshape((-1, 6))

# print(goal)
# print(object_obs)


for i in range(1000):
    obs, reward, done, _ = env.step(env.action_space.sample())
    # print(obs)
    env.render()
    if done == True:
        env.reset()
```