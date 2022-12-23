# Simulation Framework

## Environments
* There are two environments, ```SocNavEnv-v0``` and ```SocNavEnv-v1```. The latter is the implementation that has the objects defined in ```utils.py```, while the former has only humans in the environment. ```SocNavEnv-v1``` is the environment that human motion modeled, has support for static and dynamic crowds.

* To make use of the environment write the following code:
```python
import socnavenv
import gym

env = gym.make("SocNavEnv-v0") # if you want to use SocNavEnv-v0
env = gym.make("SocNavEnv-v1", config="PATH_TO_CONFIG") # if you want to use SocNavEnv-v1
```

## About the environment
```SocNavEnv-v1``` is a highly customisable environment and the parameters of the environment can be controlled using the config files. You can have a look at the config files in [paper_configs/](https://github.com/robocomp/gsoc22-socnavenv/tree/main/paper_configs). Comments have been written against each parameter in the config file for better understanding. Other than the robot, the environment supports entities like plants, tables, laptops. The environment also models interactions between humans, and human-laptop. It can also contain moving crowds, and static crowds. The environment follows the OpenAI Gym format implementing the `step`, `render` and `reset` functions. The environment follows the latest Gym API (gym 0.26.2).

## Conventions
* X-axis points in the direction of zero-angle.
* The orientation field of the humans and the robot stores the angle between the X-axis of the human/robot and the X-axis of the ground frame.

## Observation Space
The observation returned when ```env.step(action)``` is called consists of the following (all in the<b> robot frame</b>):


The observation is of the type `gym.Spaces.Dict`. The dictionary has the following keys:
1. ```"goal"``` : This is a vector of shape (8,) of which the first six values represent the one-hot encoding of the robot, i.e ```[1, 0, 0, 0, 0, 0]```. The last two values represent the goal's x and y coordinates in the robot frame.

2. The other keys present in the observation are ```"humans"```, ```"plants"```, ```"laptops"```, ```"tables"``` and ```"walls"```. Every entity (human, plant, laptop, table, or wall) would have an observation vector given by the structure below:
    <table  style=text-align:center>
        <tr>
            <th colspan="6"  style=text-align:center>Encoding</th>
            <th colspan="2" style=text-align:center>Relative Coordinates</th>
            <th colspan="2" style=text-align:center>Relative Angular Details</th>
            <th style=text-align:center>Radius</th>
            <th colspan="2" style=text-align:center>Relative Speeds</th>
            <th style=text-align:center>Gaze</th>
        </tr>
        <tr>
            <td style=text-align:center>enc0</td>
            <td style=text-align:center>enc1</td>
            <td style=text-align:center>enc2</td>
            <td style=text-align:center>enc3</td>
            <td style=text-align:center>enc4</td>
            <td style=text-align:center>enc5</td>
            <td style=text-align:center>x</td>
            <td style=text-align:center>y</td>
            <th style=text-align:center>sin(theta)</th>
            <th style=text-align:center>cos(theta)</th>
            <td style=text-align:center>radius</td>
            <td style=text-align:center>linear_vel</td>
            <td style=text-align:center>angular_vel</td>
            <td style=text-align:center>gaze</td>
        </tr>
         <tr>
            <td style=text-align:center>0</td>
            <td style=text-align:center>1</td>
            <td style=text-align:center>2</td>
            <td style=text-align:center>3</td>
            <td style=text-align:center>4</td>
            <td style=text-align:center>5</td>
            <td style=text-align:center>6</td>
            <td style=text-align:center>7</td>
            <td style=text-align:center>8</td>
            <td style=text-align:center>9</td>
            <td style=text-align:center>10</td>
            <td style=text-align:center>11</td>
            <td style=text-align:center>12</td>
            <td style=text-align:center>13</td>
        </tr>
    </table>
    Details of the field values:
    
    * One hot encodings of the object.

        The one hot encodings are as follows:
        * human:  ```[0, 1, 0, 0, 0, 0]```
        * table: ```[0, 0, 1, 0, 0, 0]```
        * laptop: ```[0, 0, 0, 1, 0, 0]```
        * plant: ```[0, 0, 0, 0, 1, 0]```
        * wall: ```[0, 0, 0, 0, 0, 1]```

    * x, y coordinates relative to the robot. For rectangular shaped objects the coordinates would correspond to the center of geometry.

    * theta : The orientation with respect to the robot

    * radius: Radius of the object. Rectangular objects will contain the radius of the circle that circumscribes the rectangle

    * relative speed

    * relative angular speed is calculated by the difference in the angles across two consecutive time steps

    * gaze value: for humans, it is 1 if the robot lies in the line of sight of humans, otherwise 0. For entities other than humans, the gaze value is 0. Line of sight of the humans is decided by whether the robot lies from -gaze_angle/2 to +gaze_angle/2 in the human frame. Gaze angle can be changed by changing the `gaze_angle` parameter in the config file.

    The observation vector of the all entities of the same type would be concatenated into a single vector and that would be placed in the corresponding key in the dictionary. For example, let's say there are 4 humans, then the four vectors of shape (14,) would be concatenated to (56,) and the `"humans"` key in the observation dictionary would contain this vector.
    
## Reward Function
The terminal states that the agent could be at are the following:
1. When agent collides with any entity
2. When agent reaches the goal
3. Out of Map (This can occur only when there are no walls in the environment. The parameter `set_shape` in the config file can be kept as "no-walls" in order for the environment to not have any walls)
4. On completing a maximum number of steps in the episode.

The rewards for these situations can be controlled using the reward section in config file for the environment. The value of parameters `collision_reward`, `reach_reward`, `out_of_map_reward`, and `max_steps_reward` would be returned for cases 1, 2, 3 and 4 respectively.


When the agent is not in a terminal state, the reward function has two forms, one is using [SNGNN](https://arxiv.org/abs/2102.08863), and the other is using [CrowdNav](https://arxiv.org/abs/1809.08835) reward function. The form of reward function is decided by the parameter `use_SNGNN` in the config file. Note that `use_SNGNN` is a factoring value and hence is a float, and not a boolean value. 


Case 1 : `use_SNGNN` = 0.0:

In this case the CrowdNav reward function would be used. Additionally if the `use_distance_to_goal` parameter in the config is True, then `distance_reward_scaler`\*(previous distance to goal - current distance to goal) is also added to the reward.

Case 2: `use_SNGNN` > 0:

In this case a reward of `use_SNGNN`\*(SNGNN output's first value) + `alive_reward` is returned (`alive_reward` can be changed from the config file). Again, additionally if the `use_distance_to_goal` parameter in the config is True, then `distance_reward_scaler`\*(previous distance to goal - current distance to goal) is also added to the reward.


## Action Space
The action space consists of three components, vx, vy, and va. Here the X axis is the robot's heading direction. For differential drive robots, the component vy would be 0. All the three components take in a value between -1 and 1, which will be later mapped to the corresponding speed using the maximum set in the config file.

## Environment Features
As mentioned, the envionment is highly configurable, and can be controlled using the config file that is passed in the `gym.make` command. These are some other features of the environment that can be controlled.

### Padded Observations:
```set_padded_observations(val:bool)```: This function is used to control the nature of the observations returned. Setting it to `True` would pad the observations. For example if the maximum possible humans are 8, but in the current scenario there are only 5 humans, then without padding the observation length for the `"humans"` key in the dictionary would be 14\*5 = 70. However, if padding was enabled, then the observation length would be 14\*8 = 112. The last 42 entries would be 0s. You can control the padding in two ways - changing the config file's `get_padded_observations` parameter, or use the environment's function as shown below:

    ```python
    import gym
    import socnavenv
    env = gym.make("SocNavEnv-v1", "configs/temp.yaml")

    env.set_padded_observations(True)
    obs, _ = env.reset() # this observation would be padded

    env.set_padded_observations(False)
    obs, _ = env.reset() # this observation would not be padded
    ```
    
### Crowd Dispersal:
The crowds that are there in the environment can also be dispersed with a specified probability. The environment would randomly disperse one of the crowds every episode (with the probability specified in the config file's `crowd_dispersal_probability` parameter)

### Supports Holonomic and Differential-Drive robots:
The environment also has support for differential-drive and holonomic robots. The parameter `robot_type` in the config file can be used to control this. The parameter takes only two values : "holonomic" or "diff-drive".

### Human Motion:
There are two models that are used to model human motion : 
1) Social Force Model (SFM)
2) Optimal Reciprocal Collision Avoidance (ORCA)

Each human would have one of these two as its policy. Whether the policy should be SFM, ORCA or randomly any one of SFM or ORCA, can be controlled using the `human_policy` parameter in the config. The parameters of both the models are randomly sampled from a Gaussian distribution. Humans can also take the robot into consideration while moving (this would make the human avoid the robot for that time step), with a probability. This can be set in by changing the `prob_to_avoid_robot` parameter in the config file. The frame of view of the humans can also be controlled using the `fov_angle` parameter.

## Wrappers
There are 4 wrappers in the environment:
1. DiscreteActions : To change the environment from a continuous action space environment to a discrete action space environment.
2. NoisyObservations : To add noise to the observations so as to emulate real world sensor noise.
3. PartialObservations : Given a fov_angle for the robot, this wrapper would only return the observations of the entities that the robot can see.
4. WorldFrameObservations : Returns all the observations in the world frame. 

