import sys
sys.path.insert(0, ".")
import gym
import torch
import socnavgym
import numpy as np
from socnavgym.envs.socnavenv_v1 import SocNavEnv_v1
from socnavgym.envs.utils import Robot, Human, Object
from socnavgym.envs.utils.utils import get_nearest_point_from_rectangle

def get_interaction_force(robot:Robot, other_human):
    e_ij = np.array([other_human.x - robot.x, other_human.y - robot.y])
    if np.linalg.norm(e_ij) != 0:
        e_ij /= np.linalg.norm(e_ij)

    robot_vx = robot.vel_x * np.cos(robot.orientation) + robot.vel_y * np.cos(np.pi/2 + robot.orientation)
    robot_vy = robot.vel_x * np.sin(robot.orientation) + robot.vel_y * np.sin(np.pi/2 + robot.orientation)

    v_ij = np.array([
        (other_human.speed * np.cos(other_human.orientation)) - robot_vx,
        (other_human.speed * np.sin(other_human.orientation)) - robot_vy
    ])

    D_ij = env.sfm_lambd *  v_ij + e_ij
    B = np.linalg.norm(D_ij) * env.sfm_gamma
    if np.linalg.norm(D_ij) != 0:
        t_ij = D_ij/np.linalg.norm(D_ij)
    theta_ij = np.arccos(np.clip(np.dot(e_ij, t_ij), -1, 1))
    n_ij = np.array([-e_ij[1], e_ij[0]])
    d_ij = np.sqrt((robot.x-other_human.x)**2 + (robot.y-other_human.y)**2)
    f_ij = -np.exp(-d_ij/B) * (np.exp(-((env.sfm_n_prime*B*theta_ij)**2))*t_ij + np.exp(-((env.sfm_n*B*theta_ij)**2))*n_ij)
    return f_ij

def get_obstacle_force(robot:Robot, obstacle:Object, r0):
        # perpendicular distance
        distance = 0

        if obstacle.name == "plant" or obstacle.name=="robot":
            distance = np.sqrt((obstacle.x - robot.x)**2 + (obstacle.y - robot.y)**2) - obstacle.radius - robot.radius
            e_o = np.array([robot.x - obstacle.x, robot.y - obstacle.y])
            if np.linalg.norm(e_o) != 0:
                e_o /= np.linalg.norm(e_o)
        
        elif obstacle.name == "human-human-interaction":
            distance = np.sqrt((obstacle.x - robot.x)**2 + (obstacle.y - robot.y)**2) - obstacle.radius - robot.radius
            e_o = np.array([robot.x - obstacle.x, robot.y - obstacle.y])
            if np.linalg.norm(e_o) != 0:
                e_o /= np.linalg.norm(e_o)
        
        elif obstacle.name == "table" or obstacle.name == "laptop":
            px, py = get_nearest_point_from_rectangle(obstacle.x, obstacle.y, obstacle.length, obstacle.width, obstacle.orientation, robot.x, robot.y)      
            e_o = np.array([robot.x - px, robot.y - py])
            if np.linalg.norm(e_o) != 0:
                e_o /= np.linalg.norm(e_o)
            distance = np.sqrt((robot.x-px)**2 + (robot.y-py)**2) - robot.radius
        
        elif obstacle.name == "wall":
            px, py = get_nearest_point_from_rectangle(obstacle.x, obstacle.y, obstacle.length, obstacle.thickness, obstacle.orientation, robot.x, robot.y)      
            e_o = np.array([robot.x - px, robot.y - py])
            if np.linalg.norm(e_o) != 0:
                e_o /= np.linalg.norm(e_o)
            distance = np.sqrt((robot.x-px)**2 + (robot.y-py)**2) - robot.radius

        else : 
            raise NotImplementedError

        f_o = np.exp(-distance/r0) * e_o
        return f_o


def compute_sfm_velocity(env:SocNavEnv_v1, robot:Robot, w1=1/np.sqrt(3), w2=1/np.sqrt(3), w3=1/np.sqrt(3)):
    f = np.array([0, 0], dtype=np.float32)
    f_d = np.zeros(2, dtype=np.float32)
    e_d = np.array([(robot.goal_x - robot.x), (robot.goal_y - robot.y)], dtype=np.float32)
    if np.linalg.norm(e_d) != 0:
        e_d /= np.linalg.norm(e_d)
    f_d = env.MAX_ADVANCE_ROBOT * e_d
    f += w1*f_d

    visible_humans = []
    visible_tables = []
    visible_plants = []
    visible_laptops = []
    visible_h_l_interactions = []
    visible_moving_interactions = []
    visible_static_interactions = []
    # walls are always visible to the human
    visible_walls = []

    # fill in the visible entities
    ## since all entities are visible to the robot, we add all of them to the visible lists
    for h in env.static_humans + env.dynamic_humans:
        visible_humans.append(h)
    
    for plant in env.plants:
        visible_plants.append(plant)
    
    for table in env.tables:
        visible_tables.append(table)
    
    for laptop in env.laptops:
        visible_laptops.append(laptop)

    for wall in env.walls:
        visible_walls.append(wall)
    
    for interaction in env.moving_interactions:
        visible_moving_interactions.append(interaction)
    
    for interaction in env.static_interactions:
        visible_static_interactions.append(interaction)
    
    for interaction in env.h_l_interactions:
        visible_h_l_interactions.append(interaction)

    for obj in visible_plants + visible_walls + visible_tables + visible_laptops + visible_static_interactions:
        f += w2 * get_obstacle_force(robot, obj, env.sfm_r0)

    for other_human in visible_humans:
        f += w3 * get_interaction_force(robot, other_human)

    for i in (visible_moving_interactions + visible_h_l_interactions):
        if i.name == "human-human-interaction":
            for other_human in i.humans:
                f += w3 * get_interaction_force(robot, other_human)

        elif i.name == "human-laptop-interaction":
            f += w3 * get_interaction_force(robot, i.human)
    
    velocity = (f/robot.mass) * env.TIMESTEP
    if np.linalg.norm(velocity) > env.MAX_ADVANCE_ROBOT:
        if np.linalg.norm(velocity) != 0:
            velocity /= np.linalg.norm(velocity)
        velocity *= env.MAX_ADVANCE_ROBOT

    return velocity

env:SocNavEnv_v1 = gym.make("SocNavGym-v1", config="configs/sngnn.yaml")
env.reset()

done = False
while not done:
    vel = compute_sfm_velocity(env, env.robot)
    vel_x = vel[0] * np.cos(env.robot.orientation) + vel[1] * np.sin(env.robot.orientation)
    vel_y = -vel[0] * np.sin(env.robot.orientation) + vel[1] * np.cos(env.robot.orientation)
    vel_a = (np.arctan2(vel[1], vel[0]) - env.robot.orientation)/env.TIMESTEP
    
    act0 = vel_x/env.MAX_ADVANCE_ROBOT
    act1 = vel_y/env.MAX_ADVANCE_ROBOT
    act2 = vel_a/env.MAX_ROTATION

    _, _, terminated, truncated, _ = env.step([act0, act1, act2])
    env.render()
    done = terminated or truncated