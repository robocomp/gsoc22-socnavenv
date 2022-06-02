"""Utility functions"""

import numpy as np
import random

def w2px(x, PIXEL_TO_WORLD, MAP_SIZE):
    """
    Given x-coordinate in world frame, to get the x-coordinate in the image frame
    """
    return int(PIXEL_TO_WORLD * (x + (MAP_SIZE / 2)))


def w2py(y, PIXEL_TO_WORLD, MAP_SIZE):
    """
    Given y-coordinate in world frame, to get the y-coordinate in the image frame
    """
    return int(PIXEL_TO_WORLD * ((MAP_SIZE / 2) - y))

def get_coordinates_of_rotated_rectangle(x, y, orientation, length, width):
    """
    Gives the coordinates of the endpoints of a rectangle centered at (x, y) and has an orientation (given by orientation)
    Returns as a list
    """
    p1 = (
        (
            x + length / 2 * np.cos(orientation)- width / 2 * np.sin(orientation),
            y + length / 2 * np.sin(orientation)+ width / 2 * np.cos(orientation)
        )
    )

    p2 = (
        (
            x + length / 2 * np.cos(orientation)+ width / 2 * np.sin(orientation),
            y + length / 2 * np.sin(orientation)- width / 2 * np.cos(orientation)
        )
    )

    p3 = (
        (
            x - length / 2 * np.cos(orientation)+ width / 2 * np.sin(orientation),
            y - length / 2 * np.sin(orientation)- width / 2 * np.cos(orientation)
        )
    )

    p4 = (
        (
            x - length / 2 * np.cos(orientation)- width / 2 * np.sin(orientation),
            y - length / 2 * np.sin(orientation)+ width / 2 * np.cos(orientation)
        )
    )

    return [p1, p2, p3, p4]

def get_coordinates_of_rotated_line(x, y, orientation, length):
    """
    Gives the coordinates of the endpoints of a line centered at (x, y) and has an orientation (given by orientation)
    Returns as a list
    """
    p1 = (
        x + (length/2)*np.cos(orientation),
        y + (length/2)*np.sin(orientation)
    )

    p2 = (
        x - (length/2)*np.cos(orientation),
        y - (length/2)*np.sin(orientation)
    )

    return [p2, p1]

def uniform_circular_sampler(center_x, center_y, radius):
    """
    For sampling uniformly in a circle with center at (center_x, center_y) and radius given by radius.
    """
    theta = 2 * np.pi * random.random()
    u = random.random() * radius
    point = (center_x + u*np.cos(theta), center_y + u*np.sin(theta))
    return point