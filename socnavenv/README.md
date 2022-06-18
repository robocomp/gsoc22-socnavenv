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

## About the environment
The environment follows the OpenAI Gym format implementing the `step`, `render` and `reset` functions. The entities other than the robot are humans, tables, laptops, plants, and walls. The room shape can be of three types namely "square", "rectangle", and "L".
* step: 

    Usage
    ```
    env.step(action)
    ```
    Here the action is from the action space (defined below).
    
    This method is used to take a step in the current episode. This function returns the next_state, reward, done, and an info dict. It has its usual meaning as any other gym environment.

    Reward function:

    The environment's reward function is as follows:
    * If the robot collides, a reward of -1 is returned, ending the episode.
    * If the robot reaches the goal, then a reward of +1 is returned, ending the episode.
    * If the episode gets over due to maximum steps then a reward of -0.5 is returned.
    * Otherwise, the reward is given by -(distance_from_goal/1000)

* render:

    Usage
    ```
    env.render()
    ```
    To visualize the environment, OpenCV has been used to give the top view of the scenario.  

* reset:

    Usage:
    ```
    env.reset()
    ```
    The scenario is reset. Firstly the shape is randomly selected, and then the number of objects of each type are randomly chosen (except for the walls). Laptops are sampled on tables, while the other objects are placed in such a way that none of them collide with each other. The goal point is also randomly sampled.


## Conventions
* X-axis points in the direction of zero-angle.
* The angle which is stored in the orientation of the humans and the robot is the angle between the X-axis of the human/robot and the X-axis of the ground frame.

## Observation Space
The observation returned when ```env.step(action)``` is called consists of the following (all in the<b> robot frame</b>):


The observation comes as a dictionary. The dictionary has the following keys:
1. ```goal``` : This contains the one-hot encoding of the robot ```[1, 0, 0, 0, 0, 0]```, which is a 6 dimensional vector. Following this, the goal's x and y coordinates are also present in the observaion space.

2. The other keys are "humans", "plants", "laptops" and "tables and the values for these have the same structure. It contains the following:
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

    * relative angular speed

    All of the above quantities would be calculated for each object in the scenario. Then, they will be concatenated with the feature vectors of objects of the same type. So, all the humans' feature vectors would be concatenated into one vector, and so on. The final 1 dimensional vectors would be stored as values in the dictionary.

## Action Space
The action space consists of the following two velocities that are given:
1. Linear Velocity
2. Angular Velocity

Both the values lie between [-1, 1]. The environment would later map these velocities to the allowed values.


## Environment Variables and Methods
1. ```set_padded_observations(val:bool)```: This function is used to control the nature of the observations returned. Setting it to `True` would pad the observations. For example if the maximum possible humans are 8, but in the current scenario there are only 5 humans, then without padding the observation length would be 13\*5 = 65. However, if padding was enabled, then the observation length would 13\*8 = 104. The last 39 entries would be 0s.

    Usage
    ```python
    import gym
    import socnavenv
    env = gym.make("SocNavEnv-v1")

    env.set_padded_observations(True)
    obs = env.reset() # this observation would be padded

    env.set_padded_observations(False)
    obs = env.reset() # this observation would not be padded
    ```
2. To get the shape of the observation space in the current scenario (if padding is enabled, then the shape will remain the same for any scenario) :
    
    Usage
    ```python
    import gym
    import socnavenv
    env = gym.make("SocNavEnv-v1")


    print(env.observation_space["humans"].shape[0]) # length of the human feature vector 
    print(env.observation_space["goal"].shape[0]) # length of the goal feature vector 
    print(env.observation_space["tables"].shape[0]) # length of the table feature vector 
    print(env.observation_space["plants"].shape[0]) # length of the plant feature vector 
    print(env.observation_space["laptops"].shape[0]) # length of the laptop feature vector 
    ```

3. ```MAX_<OBJECTS>``` where ```OBJECTS``` can be one of `HUMANS`, `TABLES`, `PLANTS` or `LAPTOPS`. This gives the maximum no. of objects for each type of 
object. 

    By default, MAX_HUMANS = 8, MAX_TABLES = 3, MAX_PLANTS = 5, MAX_LAPTOPS = 4.

    Usage:

    ```python
    import gym
    import socnavenv
    env = gym.make("SocNavEnv-v1")

    print(env.MAX_HUMANS)
    print(env.MAX_TABLES)
    print(env.MAX_PLANTS)
    print(env.MAX_LAPTOPS)
    ```

    
