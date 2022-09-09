import time
import gym
import numpy as np
import socnavenv
import os
import pygame
import numpy as np 
import sys
import argparse
import time
import pickle

os.environ['PYQTGRAPH_QT_LIB'] = 'PySide2'
from PySide2 import QtWidgets
import pyqtgraph as pg

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num_episodes", required=True, help="number of episodes")
ap.add_argument("-j", "--joystick_id", required=False, default=0, help="Joystick identifier")
args = vars(ap.parse_args())
episodes = int(args["num_episodes"])

pygame.init()
pygame.joystick.init()
joystick_count = pygame.joystick.get_count()
joystick = pygame.joystick.Joystick(int(args["joystick_id"]))
joystick.init()
axes = joystick.get_numaxes()


app = QtWidgets.QApplication(sys.argv)
plt = pg.plot()
# plt.setLogMode(y=True)
pg.setConfigOption('foreground', (0, 0, 0))
my_brush = pg.mkBrush('k', width=3)
default_brush = plt.foregroundBrush()
plt.setWindowTitle('rewards')
plt.setForegroundBrush(my_brush)
plt.addLegend()
plt.setForegroundBrush(default_brush)
plt.show()
plt.setBackground((200, 200, 200))
time.sleep(2)

try:
    with open('joystick_calibration.pickle', 'rb') as f:
        centre, values, min_values, max_values = pickle.load(f)
except:
    centre = {}
    values = {}
    min_values = {}
    max_values = {}
    for axis in range(joystick.get_numaxes()):
        values[axis] = 0.
        centre[axis] = 0.
        min_values[axis] = 0.
        max_values[axis] = 0.
    T = 3.
    print(f'Leave the controller neutral for {T} seconds')
    t = time.time()
    while time.time() - t < T:
        pygame.event.pump()
        for axis in range(axes):
            centre[axis] = joystick.get_axis(axis)
        time.sleep(0.05)
    T = 5.
    print(f'Move the joystick around for {T} seconds trying to reach the max and min values for the axes')
    t = time.time()
    while time.time() - t < T:
        pygame.event.pump()
        for axis in range(axes):
            value = joystick.get_axis(axis)-centre[axis]
            if value > max_values[axis]:
                max_values[axis] = value
            if value < min_values[axis]:
                min_values[axis] = value
        time.sleep(0.05)
    with open('joystick_calibration.pickle', 'wb') as f:
        pickle.dump([centre, values, min_values, max_values], f)
print(min_values)
print(max_values)



env = gym.make("SocNavEnv-v1")
env.configure("./configs/empty.yaml")
env.reset()
env.render()

total_sums = []

for episode in range(episodes):
    done = False

    step = -1
    prev_sum = 0
    x = []
    sums = []
    rewards = []
    sngnn = []
    while not done:
        step += 1
        plt.clear()
        pygame.event.pump()
        for i in range(joystick_count):
            print(f'{i} ', end='')
            joystick = pygame.joystick.Joystick(int(args["joystick_id"]))
            axes = joystick.get_numaxes()
            for axis in range(axes):
                values[axis] = joystick.get_axis(axis)-centre[axis]

        values[1] = max(-values[1], 0.)
        forward_speed = (values[1]-0.5)*2/max_values[1]
        angular_speed = -values[2]/max_values[2]

        obs, rew, done, info = env.step([forward_speed, angular_speed])
        print(step, forward_speed, angular_speed, info['sngnn_reward'])
        x.append(step)
        rewards.append(rew)
        sngnn.append(info['sngnn_reward'])
        new_sum = prev_sum + rew
        sums.append(new_sum)
        prev_sum = new_sum
        app.processEvents()
        env.render()

        if done:
            for _ in range(10):
                x.append(step)
                rewards.append(rew)
                sngnn.append(info['sngnn_reward'])
                sums.append(sums[-1])

        plt.plot(x, rewards, pen=pg.mkPen((255, 0, 0), width=2), name='rewards')
        plt.plot(x, sngnn, pen=pg.mkPen((0, 230, 0), width=2), name='sngnn')
        plt.plot(x, sums, pen=pg.mkPen((0, 0, 150), width=2), name='SUM')

        if done:
            for _ in range(10000):
                app.processEvents()
                time.sleep(0.01)
        app.processEvents()
    
