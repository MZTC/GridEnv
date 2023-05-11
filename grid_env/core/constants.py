from matplotlib import colors
from enum import IntEnum

COLOR_MAP = colors.ListedColormap(['white', 'black', 'green', 'red', 'blue'])

COLORS = {
    "agent": (255, 0, 0),
    "path": (0, 255, 0),
    "obstacle": (0, 0, 0),
    "none": (255, 255, 255),
    "obs": (194, 194, 194),
    "goal": (0, 0, 255)
}


class ELEMENTS(IntEnum):
    obstacle = 1
    agent = 3
    goal = 2


