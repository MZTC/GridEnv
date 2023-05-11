import matplotlib.pyplot as plt

from grid_env.core.constants import COLOR_MAP


def show_cost_map(cost_map):
    """
    绘制代价地图
    :param cost_map:
    :return:
    """
    plt.imshow(cost_map, cmap=plt.cm.hot, interpolation="nearest", vmin=0, vmax=110)


def show_grid(grid):
    """
    绘制栅格地图
    :param grid:
    :return:
    """
    plt.imshow(grid, cmap=COLOR_MAP, interpolation='nearest', vmin=0, vmax=5)
