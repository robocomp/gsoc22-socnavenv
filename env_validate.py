import gym
import socnavenv
import numpy as np
from colorama import Fore, Back, Style
from socnavenv.envs.utils.human import Human
from socnavenv.envs.utils.robot import Robot
from socnavenv.envs.utils.plant import Plant
from socnavenv.envs.utils.table import Table
from socnavenv.envs.utils.laptop import Laptop
from socnavenv.envs.utils.wall import Wall
from env_checker import check_env

env = gym.make("SocNavEnv-v1")

def empty_env(env, include_walls:bool=False):
    # humans in the environment
    env.humans.clear()
    # laptops in the environment
    env.laptops.clear() 

    if include_walls:
        # walls in the environment
        env.walls.clear() 

    # plants in the environment
    env.plants.clear() 
    # tables in the environment
    env.tables.clear()

    # interactions in the envionment
    env.interactions.clear()

env.set_padded_observations(False)

passed = 0
failed = 0
env_check = ""
observation_check = []

print("Checking environment return values")
for i in range(50):
    try:
        check_env(env)
    except Exception as e:
        env_check = (Fore.RED + "Failed Environment Return Value Check! " + f"{e}")
        failed += 1
        break

if not failed:
    env_check = Fore.GREEN + f"Passed Environment Return Value Check"


"""
Observations
Action update
"""    

print("Checking observations returned")
env.reset()
empty_env(env, True)
env.set_padded_observations(False)
env.robot.x = 5
env.robot.y = 0
env.robot.orientation = 0
env.humans.append(Human(8, 0, 0, 0.2, 0))

obs, _, done, _ = env.step([-1, 0])

try:
    assert obs["humans"].shape == (13,)
    passed += 1
    observation_check.append(Fore.GREEN + "Observation length check passed")


except AssertionError as e:
    failed += 1
    observation_check.append(Fore.RED + "Observation length check failed. " + f"{e}")

robot = [[5, 0, 0], [1, 1, 0], [0, 0, -np.pi/6], [-3, 4, np.pi/4]]
human = [[8, 0, 0], [5, 5, 0], [-np.sqrt(3), 1, 0], [-7, 0, 0]]
answer = [[3, 0, 0, 1], [4, 4, 0, 1], [-2, 0, 0.5, np.sqrt(3)/2], [-4*np.sqrt(2), 0, -1/np.sqrt(2), 1/np.sqrt(2)]]

flag = 1
for i in range(len(robot)):
    env.robot.x = robot[i][0]
    env.robot.y = robot[i][1]
    env.robot.orientation = robot[i][2]

    env.humans[0].x = human[i][0]
    env.humans[0].y = human[i][1]
    env.humans[0].orientation = human[i][2]
    
    obs, _, done, _ = env.step([-1, 0])
    obs = obs["humans"]
    if (np.abs(obs[6]-answer[i][0])<=1e-6) and (np.abs(obs[7]-answer[i][1])<=1e-6) and (np.abs(obs[8]-answer[i][2])<=1e-6) and (np.abs(obs[9]-answer[i][3])<=1e-6):
        passed += 1
    else:
        failed += 1
        observation_check.append(Fore.RED + f"Failed check {i}" + f"\nAnswer : {answer[i][0]} {answer[i][1]} {answer[i][2]} {answer[i][3]}, Got {obs[6]} {obs[7]} {obs[8]} {obs[9]}")
        flag = 0

    if done:
        env.reset()
        empty_env(env, True)
        env.set_padded_observations(False)
        env.robot.x = 5
        env.robot.y = 0
        env.robot.orientation = 0
        env.humans.append(Human(8, 0, 0, 0.2, 0))

if flag:
    observation_check.append(Fore.GREEN + "Passed Observation checks")

env.reset()
empty_env(env, True)
env.set_padded_observations(False)
env.robot.x = 5
env.robot.y = 0
env.robot.orientation = 0
env.humans.append(Human(8, 0, 0, 0.2, 0.1))


action_check = []
print("Checking action updates")

env.step([1,0.5])
if env.humans[0].x == 8.1 and env.humans[0].y == 0: 
    passed+=1
    action_check.append(Fore.GREEN + "Human action check passed")
else:
    failed += 1
    action_check.append(Fore.RED + "Human action check failed")
if np.abs(env.robot.x-5) <= 1e-6 and np.abs(env.robot.y - 0.1) <= 1e-6 and np.abs(env.robot.orientation - np.pi/2) <= 1e-6: 
    passed+=1
    action_check.append(Fore.GREEN + "Robot action check passed")
else:
    failed += 1
    action_check.append(Fore.RED + "Robot action check failed")


print(env_check)
for v in observation_check: print(v)
for v in action_check: print(v)

print(f"Total tests: {passed+failed}" + Fore.GREEN + f" Passed: {passed} ", end=" ")
s = (Fore.RED+f" Failed: {failed}") if failed else (Fore.GREEN + ":)")
print(s)