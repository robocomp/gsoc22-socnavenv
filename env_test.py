import gym
import socnavenv
import cv2
import numpy as np
from socnavenv.envs.utils import *


RESOLUTION = 700.0
RESOLUTION_VIEW = 1000.0
MAP_SIZE = 8.0
PIXEL_TO_WORLD = RESOLUTION / MAP_SIZE
cv2.namedWindow("world", cv2.WINDOW_NORMAL)
cv2.resizeWindow("world", int(RESOLUTION_VIEW), int(RESOLUTION_VIEW))


r = Robot(1, 0, -np.pi, 0.4)
h1 = Human(0, 3, 5*np.pi/4, 0.75, 0.01)
h1.set_color((12, 84, 65))
h2 = Human(2, -3, -5*np.pi/4, 0.75, 0.01)
h2.set_color((78, 87, 211))
t = Table(2, 2, np.pi/2, 3, 2)
p1 = Plant(-3, -3, 0.2)
p2 = Plant(-3, 3, 0.2)
l = Laptop(2, 2, np.pi/2, 0.35, 0.4)
w1 = Wall(0, 4, 0, 8)
w2 = Wall(4, 0, np.pi/2, 8)
w3 = Wall(0, -4, 0, 8)
w4 = Wall(-4, 0, np.pi/2, 8)

le = [r, h1, h2, t, p1, p2, l, w1, w2 ,w3 ,w4]

time = 0.25
for i in range(1000):
    r.update(0.02, 0.01, time)
    h1.update(time)
    h2.update(time)
    img = (np.ones((int(RESOLUTION), int(RESOLUTION), 3)) * 255).astype(np.uint8)
    for obj in le:
        obj.draw(img, PIXEL_TO_WORLD, MAP_SIZE)

    cv2.imshow("world", img)
    cv2.waitKey(3)
cv2.destroyAllWindows()