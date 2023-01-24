# SocNavEnv : An environment for Social Navigation

## Description

Socially aware path planning enables a robot to navigate through a crowded environment causing the least amount of discomfort to the surrounding people. Building upon the work done in GSoC '21, the aim of the project is to account for the interactions between the different entities in the environment by adding interaction nodes between entities in the graph representation. Since the simulators available are heavy to train, a light-weight OpenAI Gym style environment has been designed. The details of the environment can be found [here](https://github.com/robocomp/gsoc22-socnavenv/tree/main/socnavenv). 
Deep RL agents have also been implemented in the [agents](https://github.com/robocomp/gsoc22-socnavenv/tree/main/agents) directory that contains implementations of DQN, DuelingDQN, A2C, PPO, DDPG, SAC, and CrowdNav, using a custom transformer as the underlying architecture.

## Dependencies
The following dependencies can be installed using pip or Anaconda: `gym` `matplotlib` `opencv-python`, `shapely`, `Cython`, `cv2`.
To run a few agents, `stable-baselines-3` is required. It can be installed using : `pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests`

RVO2 can be installed using the following repository: https://github.com/sybrenstuvel/Python-RVO2/

## Usage
```python
import socnavenv
import gym
env = gym.make('SocNavEnv-v1', config="<PATH_TO_CONFIG>")  
```
## Sample Code
```python
import socnavenv
import gym
env = gym.make("SocNavEnv-v1", config="./configs/temp.yaml") 
obs, _ = env.reset()


for i in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.render()
    if terminated or truncated:
        env.reset()
```

## Training Agents:
```python
python3 train.py -a="agent_name" -t="transformer/mlp" -e="path_to_env_config" -c="path_to_agent_config"
```
