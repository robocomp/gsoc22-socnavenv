import time
import gym
import numpy as np
import socnavenv
import os
import pygame
import numpy as np 
import matplotlib.pyplot as plt
import sys
import argparse

pygame.init()
pygame.key.set_repeat(50)
display = pygame.display.set_mode((1,1))
env = gym.make("SocNavEnv-v1")
env.configure("./configs/empty.yaml")
env.reset()
env.render()

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num_episodes", required=True, help="number of episodes")
args = vars(ap.parse_args())

episodes = int(args["num_episodes"])
total_sums = []

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

for episode in range(episodes):
    done = False

    rewards = []
    forward_vel = -1
    angular_vel = 0
    while not done:
        flag1 = 0
        flag2 = 0
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if(pygame.key.name(event.key) == "up"):
                    forward_vel += 0.2
                    forward_vel = min(forward_vel, 1.0)
                    flag1 = 1
                
                elif(pygame.key.name(event.key) == "down"):
                    forward_vel -= 0.2
                    forward_vel = max(forward_vel, -1.0)
                    flag1 = 1
                
                elif(pygame.key.name(event.key) == "right"):
                    angular_vel -= 0.005
                    angular_vel = min(angular_vel, 1.0)
                    flag2 = 1
                
                elif(pygame.key.name(event.key) == "left"):
                    angular_vel += 0.005
                    angular_vel = max(angular_vel, -1.0)
                    flag2 = 1
        
        if not flag1:
            forward_vel -= 0.3
            forward_vel = max(forward_vel, -1.0)

        if not flag2:
            if angular_vel >= 0:
                angular_vel -= 0.01
                angular_vel = max(0, angular_vel)
            else:
                angular_vel += 0.01
                angular_vel = min(0, angular_vel)
        time.sleep(0.05)
        obs, rew, done, _ = env.step([forward_vel, angular_vel])
        rewards.append(rew)
        env.render()
        
    
    x = [i for i in range(1, len(rewards)+1)]
    sum = 0
    for i in rewards: sum += i
    total_sums.append(sum)
    ax1.plot(x, rewards, label=f"episode {episode}")
    obs = env.reset()


ax1.legend()
ax1.set_title("Rewards / time step")
ax1.set_xlabel("time")
ax1.set_ylabel("reward")

ax2.plot([i+1 for i in range(len(total_sums))], total_sums)
ax2.set_title("total rewards / episode")
ax2.set_xlabel("episode")
ax2.set_ylabel("reward")

plt.show()