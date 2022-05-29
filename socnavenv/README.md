# Simulation Framework

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