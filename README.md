# SocNavGym : An environment for Social Navigation

## Description

Socially aware path planning enables a robot to navigate through a crowded environment causing the least amount of discomfort to the surrounding people. The aim of the project is to account for the interactions between the different entities in the environment by adding interaction nodes between entities in the graph representation. Since the simulators available are heavy to train, a light-weight OpenAI Gym style environment has been designed. Additionally, 
Deep RL agents have also been implemented in the [agents](https://github.com/robocomp/gsoc22-socnavenv/tree/main/agents) directory that contains implementations of DQN, DuelingDQN, A2C, PPO, DDPG, SAC, and CrowdNav, using a custom transformer as the underlying architecture. Scripts usign Stable Baselines 3 are also available (`stable_dqn.py` and `stable_ppo.py`). 

## Dependencies
The following dependencies can be installed using pip or Anaconda: `gym` `matplotlib` `opencv-python`, `shapely`, `Cython`, `cv2`.
To run a few agents, `stable-baselines-3` is required. It can be installed using : `pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests`

RVO2 can be installed using the following repository: https://github.com/sybrenstuvel/Python-RVO2/

## Usage
```python
import socnavgym
import gym
env = gym.make('SocNavGym-v1', config="<PATH_TO_CONFIG>")  
```
## Sample Code
```python
import socnavgym
import gym
env = gym.make("SocNavGym-v1", config="./configs/temp.yaml") 
obs, _ = env.reset()


for i in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.render()
    if terminated or truncated:
        env.reset()
```

## About the environment
```SocNavGym-v1``` is a highly customisable environment and the parameters of the environment can be controlled using the config files. You can have a look at the config files in [paper_configs/](https://github.com/robocomp/gsoc22-socnavenv/tree/main/paper_configs). Comments have been written against each parameter in the config file for better understanding. Other than the robot, the environment supports entities like plants, tables, laptops. The environment also models interactions between humans, and human-laptop. It can also contain moving crowds, and static crowds. The environment follows the OpenAI Gym format implementing the `step`, `render` and `reset` functions. The environment uses the latest Gym API (gym 0.26.2).

## Conventions
* X-axis points in the direction of zero-angle.
* The orientation field of the humans and the robot stores the angle between the X-axis of the human/robot and the X-axis of the ground frame.

## Observation Space
The observation returned when ```env.step(action)``` is called consists of the following (all in the<b> robot frame</b>):


The observation is of the type `gym.Spaces.Dict`. The dictionary has the following keys:
1. ```"robot"``` : This is a vector of shape (9,) of which the first six values represent the one-hot encoding of the robot, i.e ```[1, 0, 0, 0, 0, 0]```. The next two values represent the goal's x and y coordinates in the robot frame and the last value is the robot's radius.

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
            <td style=text-align:center>sin(theta)</th>
            <td style=text-align:center>cos(theta)</th>
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

    * relative angular speed is calculated by the difference in the angles across two consecutive time steps and dividing by the time-step

    * gaze value: for humans, it is 1 if the robot lies in the line of sight of humans, otherwise 0. For entities other than humans, the gaze value is 0. Line of sight of the humans is decided by whether the robot lies from -gaze_angle/2 to +gaze_angle/2 in the human frame. Gaze angle can be changed by changing the `gaze_angle` parameter in the config file.

    The observation vector of the all entities of the same type would be concatenated into a single vector and that would be placed in the corresponding key in the dictionary. For example, let's say there are 4 humans, then the four vectors of shape (14,) would be concatenated to (56,) and the `"humans"` key in the observation dictionary would contain this vector. Individual observations can be accessed by simply reshaping the observation to (-1, 14).

    For walls, each wall is segmented into smaller walls of size `wall_segment_size` from the config. Observations from each segment are returned in `obs["walls"]`

## Action Space
The action space consists of three components, vx, vy, and va. Here the X axis is the robot's heading direction. For differential drive robots, the component vy would be 0. All the three components take in a value between -1 and 1, which will be later mapped to the corresponding speed using the maximum set in the config file. If you want to use a discrete action space, you could use the `DiscreteActions` wrapper.

## Info Dict
The environment also returns meaningful metrics in the info dict (returned during `env.step()` ). On termination of an episode, the reason of termination can be found out using the info dict. The corresponding key from the following keys is set to `True`.
1. `"OUT_OF_MAP"` : Whether the robot went out of the map
2. `"REACHED_GOAL"` : Whether the robot reached the goal
3. `"COLLISION_HUMAN"` : Whether the robot collided with a human
4. `"COLLISION_OBJECT"` : Whether the robot collided with an object (plant, wall, table)
5. `"COLLISION"` : Whether the robot collided with any entity
6. `"MAX_STEPS"` : Has the episode terminated because the maximum length of episode was reached.


Apart from this, the info dict also provides a few metrics such as `"personal_space_compliance"` (percentage of time that the robot is not within the personal space of humans), `"success_weighted_by_time_length"` (success * (time_taken_by_orca_agent_to_reach_goal / time_taken_by_robot_to_reach_goal)), `"closest_human_dist"` (distance to the closest human), and `"closest_obstacle_dist"` (distance to the closest obstacle). Some additional metrics that are also provided are :
* `"DISCOMFORT_SNGNN"` : SNGNN_value (More about SNGNN in the section below)
* `"DISCOMFORT_DSRNN"` : DSRNN reward value  (More about DSRNN reward function in the section below)
* `"sngnn_reward"` : SNGNN_value - 1
* `"distance_reward"` : Value of the distance reward. 
Note that the above 4 values are returned correctly if the reward function used is `"dsrnn"` or `"sngnn"`. If a custom reward function is written, then the user is required to fill the above values otherwise 0s would be returned for them.
    
Lastly, information about the interactions is returned as an adjacency list. There are two types of interactions, `"human-human"` and `"human-laptop"`. For every interaction between human `i` and human `j` (`i` and `j` are the based on the order in which the human's observations appear in the observation dictionary. So to extract the ith human's observation, you could just to `obs["humans"].reshape(-1, 14)[i]`), (i, j) would be present in `info["interactions"]["human-human"]`, and similarly for an interaction between the ith human and the jth laptop, (i, j) would be present in `info["interactions"]["human-laptop"]`.

## Reward Function
The environment provides implementation of the [SNGNN](https://arxiv.org/abs/2102.08863) reward function, and the [DSRNN](https://arxiv.org/abs/2011.04820) reward function. If you want to use these reward functions, the config passed to the environment should have the value corresponding to the field `reward_file` as `"sngnn"` or `"dsrnn"` respectively. 


The environment also allows users to provide custom reward functions. Follow the guide below to create your own reward function.

## Writing Custom Reward Functions

1. Create a new python file in which you **have to** create a class named `Reward`. It **must** inherit from `RewardAPI` class. To do this, do the following
```python
from socnavgym.envs.rewards import RewardAPI

class Reward(RewardAPI):
    ...
```
2. Overwrite the function `compute_reward` with the custom reward function. The input of the `compute_reward` function is the action of the current timestep, the previous entity observations and the current entity observations. The previous and current observations are given as a dictionary with key as the id of the entity, and the value is an instance of the `EntityObs` namedtuple defined in [this](https://github.com/robocomp/gsoc22-socnavenv/blob/main/socnavgym/envs/socnavenv_v1.py) file. It contains the fields : id, x, y, theta, sin_theta, cos_theta for each entity in the environment. Note that all these values are in the robot's frame of reference. 
3. If need be, you can also access the lists of humans, plants, interactions etc, that the environment maintains by referencing the `self.env` variable. An example of this can be found in the [`dsrnn_reward.py`](https://github.com/robocomp/gsoc22-socnavenv/blob/main/socnavgym/envs/rewards/dsrnn_reward.py) file
4. The `RewardAPI` class provides helper functions such as `check_collision`, `check_timeout`, `check_reached` and `check_out_of_map`. These functions are boolean functions that check if the robot has collided with any enitity, whether the maximum episode length has been reached, whether the robot has reached the goal, or if the robot has moved out of the map respectively. The last case can occur only when the envirnoment is configured to have no walls.
5. The `RewardAPI` class also has a helper function defined to compute the SNGNN reward function. Call `compute_sngnn_reward(actions, prev_obs, curr_obs)` to compute the SNGNN reward. Also note that if you are using the SNGNN reward function in your custom reward function, please set the variable `self.use_sngnn` to `True`.
6. You can also store any additional information that needs to be returned in the info dict of step function by storing all of it in the variable `self.info` of the `Reward` class.
7. Storing anything in a class variable will persist across the steps in an episode. There will be a new instantiation of the Reward class object every episode.
8. Provide the path to the file where you defined your custom reward function in the config file's `reward_file`.


## Config File
The behaviour of the environment is controlled using the config file. The config file needs to be passed as a parameter while doing `gym.make`. These are the following parameters and their corresponding descriptions


<table  style=text-align:center>
    <tr>
        <th style=text-align:center> </th>
        <th style=text-align:center>Parameter</th>
        <th style=text-align:center>Description</th>
    </tr>
    <tr>
        <td rowspan="2"> rendering </td>
        <td> resolution_view </td>
        <td> size of the window for rendering the environment </td>
    </tr>
    <tr>
        <td> milliseconds </td>
        <td> delay parameter for waitKey()</td>
    </tr>
    <tr>
        <td rowspan="2"> episode </td>
        <td> episode_length </td>
        <td> maximum steps in an episode</td>
    </tr>
    <tr>
        <td> time_step </td>
        <td> number of seconds that one step corresponds to</td>
    </tr>
    <tr>
        <td rowspan="3"> robot </td>
        <td> robot_radius </td>
        <td> radius of the robot</td>
    </tr>
    <tr>
        <td> goal_radius </td>
        <td> radius of the robot's goal</td>
    </tr>
    <tr>
        <td> robot_type </td>
        <td> Accepted values are "diff-drive" (for differential drive robot) and "holonomic" (for holonomic robot)</td>
    </tr>
    <tr>
        <td rowspan="6"> human </td>
        <td> human_diameter </td>
        <td> diameter of the human</td>
    </tr>
    <tr>
        <td> human_goal_radius </td>
        <td> radius of human's goal</td>
    </tr>
    <tr>
        <td> human_policy </td>
        <td> policy of the human. Can be "random", "sfm", or "orca". If "random" is kept, then one of "orca" or "sfm" would be randomly chosen</td>
    </tr>
    <tr>
        <td> gaze_angle </td>
        <td> gaze value (in the observation) for humans would be set to 1 when the robot lies between -gaze_angle/2 and +gaze_angle/2</td>
    </tr>
    <tr>
        <td> fov_angle </td>
        <td> the frame of view for humans</td>
    </tr>
    <tr>
        <td> prob_to_avoid_robot </td>
        <td> the probability that the human would consider the robot in its policy</td>
    </tr>
    <tr>
        <td rowspan="2"> laptop </td>
        <td> laptop_width </td>
        <td> width of laptops</td>
    </tr>
    <tr>
        <td> laptop_length </td>
        <td> length of laptops</td>
    </tr>
    <tr>
        <td rowspan="1"> plant </td>
        <td> plant_radius </td>
        <td> radius of plant<s/td>
    </tr>
     <tr>
        <td rowspan="2"> table </td>
        <td> table_width </td>
        <td> width of tables</td>
    </tr>
    <tr>
        <td> table_length </td>
        <td> length of tables</td>
    </tr>
    <tr>
        <td rowspan="1"> wall </td>
        <td> wall_thickness </td>
        <td> thickness of walls</td>
    </tr>
    <tr>
        <td rowspan="3"> human-human-interaction </td>
        <td> interaction_radius </td>
        <td> radius of the human-crowd</td>
    </tr>
    <tr>
        <td> interaction_goal_radius </td>
        <td> radius of the human-crowd's goal</td>
    </tr>
    <tr>
        <td> noise_varaince </td>
        <td> a random noise of normal(0, noise_variance) is applied to the humans' speed to break uniformity</td>
    </tr>
    <tr>
        <td rowspan="1"> human-laptop-interaction </td>
        <td> human_laptop_distance </td>
        <td> distance between human and laptop</td>
    </tr>
    <tr>
        <td rowspan="43"> env </td>
        <td> margin </td>
        <td> margin for the env </td>
    </tr>
    <tr>
        <td> max_advance_human </td>
        <td> maximum speed for humans </td>
    </tr>
    <tr>
        <td> max_advance_robot </td>
        <td> maximum linear speed for the robot </td>
    </tr>
    <tr>
        <td> max_rotation </td>
        <td> maximum rotational speed for robot </td>
    </tr>
    <tr>
        <td> wall_segment_size </td>
        <td> size of the wall segment, used when segmenting the wall </td>
    </tr>
    <tr>
        <td> speed_threshold </td>
        <td> speed below which would be considered 0 (for humans) </td>
    </tr>
    <tr>
        <td> crowd_dispersal_probability </td>
        <td> probability of crowd dispersal </td>
    </tr>
    <tr>
        <td> human_laptop_dispersal_probability </td>
        <td> probability to disperse a human-laptop-interaction </td>
    </tr>
    <tr>
        <td> crowd_formation_probability </td>
        <td> probability of crowd formation </td>
    </tr>
    <tr>
        <td> human_laptop_formation_probability </td>
        <td> probability to form a human-laptop-interaction </td>
    </tr>
    <tr>
        <td> reward_file </td>
        <td> Path to custom-reward file. If you want to use the in-built SNGNN reward function or the DSRNN reward function, set the value to "sngnn" or "dsrnn" respectively </td>
    </tr>
    <tr>
        <td> cuda_device </td>
        <td> cuda device to use (in case of multiple cuda devices). If cpu or only one cuda device, keep it as 0 </td>
    </tr>
    <tr>
        <td> min_static_humans </td>
        <td> minimum no. of static humans in the environment</td>
    </tr>
    <tr>
        <td> max_static_humans </td>
        <td> maximum no. of static humans in the environment</td>
    </tr>
    <tr>
        <td> min_dynamic_humans </td>
        <td> minimum no. of dynamic humans in the environment</td>
    </tr>
    <tr>
        <td> max_dynamic_humans </td>
        <td> maximum no. of dynamic humans in the environment</td>
    </tr>
    <tr>
        <td> min_tables </td>
        <td> minimum no. of tables in the environment</td>
    </tr>
    <tr>
        <td> max_tables </td>
        <td> maximum no. of tables in the environment</td>
    </tr>
    <tr>
        <td> min_plants </td>
        <td> minimum no. of plants in the environment</td>
    </tr>
    <tr>
        <td> max_plants </td>
        <td> maximum no. of plants in the environment</td>
    </tr>
    <tr>
        <td> min_laptops </td>
        <td> minimum no. of laptops in the environment</td>
    </tr>
    <tr>
        <td> max_laptops </td>
        <td> maximum no. of laptops in the environment</td>
    </tr>
    <tr>
        <td> min_h_h_dynamic_interactions </td>
        <td> minimum no. of dynamic human-human interactions in the env. Note that these crowds can disperse if the parameter crowd_dispersal_probability is greater than 0 </td>
    </tr>
    <tr>
        <td> max_h_h_dynamic_interactions </td>
        <td> maximum no. of dynamic human-human interactions in the env. Note that these crowds can disperse if the parameter crowd_dispersal_probability is greater than 0</td>
    </tr>
    <tr>
        <td> min_h_h_dynamic_interactions_non_dispersing </td>
        <td> minimum no. of dynamic human-human interactions in the env. Note that these crowds never disperse, even if the parameter crowd_dispersal_probability is greater than 0 </td>
    </tr>
    <tr>
        <td> max_h_h_dynamic_interactions_non_dispersing </td>
        <td> maximum no. of dynamic human-human interactions in the env. Note that these crowds never disperse, even if the parameter crowd_dispersal_probability is greater than 0</td>
    </tr>
    <tr>
        <td> min_h_h_static_interactions </td>
        <td> minimum no. of static human-human interactions in the env. Note that these crowds can disperse if the parameter crowd_dispersal_probability is greater than 0 </td>
    </tr>
    <tr>
        <td> max_h_h_static_interactions </td>
        <td> maximum no. of static human-human interactions in the env. Note that these crowds can disperse if the parameter crowd_dispersal_probability is greater than 0</td>
    </tr>
    <tr>
        <td> min_h_h_static_interactions_non_dispersing </td>
        <td> minimum no. of static human-human interactions in the env. Note that these crowds never disperse, even if the parameter crowd_dispersal_probability is greater than 0 </td>
    </tr>
    <tr>
        <td> max_h_h_static_interactions_non_dispersing </td>
        <td> maximum no. of static human-human interactions in the env. Note that these crowds never disperse, even if the parameter crowd_dispersal_probability is greater than 0</td>
    </tr>
    <tr>
        <td> min_human_in_h_h_interactions </td>
        <td> minimum no. of humans in a human-human interaction </td>
    </tr>
    <tr>
        <td> max_human_in_h_h_interactions </td>
        <td> maximum no. of humans in a human-human interaction </td>
    </tr>
    <tr>
        <td> min_h_l_interactions </td>
        <td> minimum no. of human-laptop interactions in the env. Note that these crowds can disperse if the parameter human_laptop_dispersal_probability is greater than 0 </td>
    </tr>
    <tr>
        <td> max_h_l_interactions </td>
        <td> maximum no. of human-laptop interactions in the env. Note that these crowds can disperse if the parameter human_laptop_dispersal_probability is greater than 0</td>
    </tr>
    <tr>
        <td> min_h_l_interactions_non_dispersing </td>
        <td> minimum no. of human-laptop interactions in the env. Note that these crowds never disperse, even if the parameter human_laptop_dispersal_probability is greater than 0 </td>
    </tr>
    <tr>
        <td> max_h_l_interactions_non_dispersing </td>
        <td> maximum no. of human-laptop interactions in the env. Note that these crowds never disperse, even if the parameter human_laptop_dispersal_probability is greater than 0</td>
    </tr>
    <tr>
        <td> get_padded_observations </td>
        <td> flag value that indicates whether you require padded observations or not. You can change it using env.set_padded_observations(True/False) </td>
    </tr>
    <tr>
        <td> set_shape </td>
        <td> Sets the shape of the environment. Accepted values are "random", "square", "rectangle", "L" or "no-walls"  </td>
    </tr>
    <tr>
        <td> add_corridors </td>
        <td> True or False, whether there should be corridors in the environment</td>
    </tr>
    <tr>
        <td> min_map_x </td>
        <td> minimum size of map along x direction </td>
    </tr>
    <tr>
        <td> max_map_x </td>
        <td> maximum size of map along x direction </td>
    </tr>
    <tr>
        <td> min_map_y </td>
        <td> minimum size of map along y direction </td>
    </tr>
    <tr>
        <td> max_map_y </td>
        <td> maximum size of map along y direction </td>
    </tr>
    
</table>

## Wrappers
Gym wrappers are convenient to have changes in the observation-space / action-space. SocNavGym implements 4 wrappers. 

The following are the wrappers implemented by SocNavGym:
1. `DiscreteActions` : To change the environment from a continuous action space environment to a discrete action space environment. The action space consists of 7 discrete actions. They are :
    * Turn anti-clockwise (0)
    * Turn clock-wise (1)
    * Turn anti-clockwise and moving forward (2)
    * Turning clockwise and moving forward (3)
    * Move forward (4)
    * Move backward (5)
    * Stay still (6)

    As an example, to make the robot move forward throughout the episode, just do the following:
    ```python
    import socnavgym
    from socnavgym.wrappers import DiscreteActions

    env = gym.make("SocNavGym-v1", config="paper_configs/exp1_no_sngnn.yaml")  # you can pass any config
    env = DiscreteActions(env)  # creates an env with discrete action space

    # simulate an episode with random actions
    done = False
    env.reset()
    while not done:
        obs, rew, terminated, truncated, info = env.step(4)  # 4 is for moving forward 
        done = terminated or truncated
        env.render()

    ```

2. `NoisyObservations` : This wrapper can be used to add noise to the observations so as to emulate real world sensor noise. The parameters that the wrapper takes in are `mean`, `std_dev`. Apart from this, there is also a parameter called `apply_noise_to` which defaults to `[robot", "humans", "tables", "laptops", "plants", "walls"]`, meaning all enitity types. If you want to apply noise to only a few entity types, then pass a list with only those entity types to this parameter. The noise value can be controlled using the `mean` and the `std_dev` parameters. Basically, a Gaussian noise with `mean` and `std_dev` is added to the observations of all the entities whose entity type is listed in the parameter `apply_noise_to`.
    As an example, to add a small noise with 0 mean and 0.1 std dev to all entity types do the following:
    ```python
    import socnavgym
    from socnavgym.wrappers import NoisyObservations

    env = gym.make("SocNavGym-v1", config="paper_configs/exp1_no_sngnn.yaml")  # you can pass any config
    env = NoisyObservations(env, mean=0, std_dev=0.1)

    # simulate an episode with random actions
    done = False
    env.reset()
    while not done:
        obs, rew, terminated, truncated, info = env.step(env.action_space.sample())  # obs would now be a noisy observation

        done = terminated or truncated
        env.render()

    ```

3. PartialObservations : This wrapper is used to return observations that are present in the frame of view of the robot, and also that lies within the range. Naturally, the wrapper takes in two parameters `fov_angle` and the `range`. 
    An example of using the `PartialObservations` wrapper:

    ```python
    import socnavgym
    from socnavgym.wrappers import PartialObservations
    from math import pi

    env = gym.make("SocNavGym-v1", config="paper_configs/exp1_no_sngnn.yaml")  # you can pass any config
    env = PartialObservations(env, fov_angle=2*pi/3, range=1)  # creates a robot with a 120 degreee frame of view, and the sensor range is 1m.

    # simulate an episode with random actions
    env.reset()
    done = False
    while not done:
        obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        env.render()

    ```
4. WorldFrameObservations : Returns all the observations in the world frame. The observation space of the `"robot"` would look like this:
    <table  style=text-align:center>
            <tr>
                <th colspan="6"  style=text-align:center>Encoding</th>
                <th colspan="2" style=text-align:center> Robot Goal coordinates</th>
                <th colspan="2" style=text-align:center> Robot coordinates</th>
                <th colspan="2" style=text-align:center> Angular Details</th>
                <th colspan="3" style=text-align:center>Velocities Speeds</th>
                <th style=text-align:center>Radius</th>
            </tr>
            <tr>
                <td style=text-align:center>enc0</td>
                <td style=text-align:center>enc1</td>
                <td style=text-align:center>enc2</td>
                <td style=text-align:center>enc3</td>
                <td style=text-align:center>enc4</td>
                <td style=text-align:center>enc5</td>
                <td style=text-align:center>goal_x</td>
                <td style=text-align:center>goal_y</td>
                <td style=text-align:center>x</td>
                <td style=text-align:center>y</td>
                <td style=text-align:center>sin(theta)</th>
                <td style=text-align:center>cos(theta)</th>
                <td style=text-align:center>vel_x</th>
                <td style=text-align:center>vel_y</th>
                <td style=text-align:center>vel_a</th>
                <td style=text-align:center>radius</td>
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
                <td style=text-align:center>14</td>
                <td style=text-align:center>15</td>
            </tr>
        </table>
    The other enitity observations would remain the same, the only difference being that the positions and velocities would be in the world frame of reference and not in the robot's frame of reference. 
    An example of using the `PartialObservations` wrapper:

    ```python
    import socnavgym
    from socnavgym.wrappers import WorldFrameObservations
    from math import pi

    env = gym.make("SocNavGym-v1", config="paper_configs/exp1_no_sngnn.yaml")  # you can pass any config
    env = WorldFrameObservations(env) 

    # simulate an episode with random actions
    env.reset()
    done = False
    while not done:
        obs, rew, terminated, truncated, info = env.step(env.action_space.sample())  # obs contains observations that are in the world frame 
        done = terminated or truncated
        env.render()

    ```
