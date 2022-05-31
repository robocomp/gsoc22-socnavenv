from abc import abstractmethod


class Object(object):
    """
    Base class for the objects in the environment.

    Objects :
    - tables
    - laptops
    - plants
    - walls
    - humans
    - robot

    """

    def __init__(self, name: str) -> None:
        self.x = None  # x coordinate (generally the x-coordinate of the center of mass)
        self.y = None  # y coordinate (generally the y-coordinate of the center of mass)
        self.orientation = None  # angle with the X-axis of the global frame
        self.name = name  # string representing the type of object
        self.is_static = (
            True  # boolean variable denoting whether the object is static or dynamic
        )
        self.color = None  # color of the object for drawing purposes

    def set(self, x, y, theta):
        """
        Method for setting attributes of the object
        """
        self.x = x
        self.y = y
        self.orientation = theta

    def set_color(self, color):
        """
        Method to set the color of the object
        """
        self.color = color

    def get_position(self):
        """
        To get the coordinates of the object
        """
        return (self.x, self.y)

    def get_orientation(self):
        """
        To get the orientation of the object
        """
        return self.orientation

    def get_name(self):
        """
        To get the name (type) of the object
        """
        return self.name

    @abstractmethod
    def draw(self, img, PIXEL_TO_WORLD, MAP_SIZE, color=None, radius=None, nose=None):
        """
        Function for drawing the object on the image.
        """
        return
