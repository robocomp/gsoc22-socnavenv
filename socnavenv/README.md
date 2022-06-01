# Simulation Framework

## Environments
* There are two environments, ```SocNavEnv-v0``` and ```SocNavEnv-v1```. The latter is the implementation that has the objects defined in ```utils.py```, while the former has only humans in the environment. 
* To make use of the environment write the following code:
```python
import socnavenv
import gym

env = gym.make("SocNavEnv-v0") # if you want to use SocNavEnv-v0
env = gym.make("SocNavEnv-v1") # if you want to use SocNavEnv-v1
```


## Conventions
* X-axis points in the direction of zero-angle.
* The angle which is stored in the orientation of the humans and the robot is the angle between the X-axis of the human/robot and the X-axis of the ground frame.

## Observation Space
The observation returned when ```env.step(action)``` is called consists of the following (all in the<b> robot frame</b>):
1. goal of the robot (x and y coordinate)
2. For each human (x, y, sin(theta), cos(theta), relative linear speed, relative angular speed). (here theta is the relative angle between the human and the robot)


## Action Space
The action space consists of the following two velocities that are given:
1. Linear Velocity
2. Angular Velocity

Both the values lie between [-1, 1]. The environment would later map these velocities to the allowed values.