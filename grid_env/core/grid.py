import matplotlib.pyplot as plt
import numpy as np

from grid_env.core.render_util import show_grid, show_cost_map
from grid_env.core.constants import ELEMENTS

X = 0
Y = 1


class Grid:
    def __init__(self, width: int, height: int, threshold=70, obstacle_num=16, emission_step=4, emission_dec=20):
        """

        :param width:
        :param height:
        """

        assert width >= 60
        assert height >= 60

        self.width: int = width
        self.height: int = height

        self.threshold = threshold
        self.emission_dec = emission_dec
        self.emission_step = emission_step
        self.obstacle_num = obstacle_num

        self.grid = np.zeros((self.height, self.width), dtype=np.int8)
        self.cost_map = np.zeros((self.height, self.width), dtype=np.int8)
        self.obstacles = []

        self.agent_pos = None
        self.goal = None

    def reset_map(self):
        """
        重新生成地图
        :return:
        """
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)
        self.cost_map = np.zeros((self.height, self.width), dtype=np.int8)
        self.obstacles = []

        self.generate_wall()
        self.generate_obstacle_rand(o_num=self.obstacle_num)

        self.cost_map[self.grid == ELEMENTS.obstacle] = 100
        self.emission(emission_step=self.emission_step, dec=self.emission_dec)

        self.goal = self.get_legal_pos()
        self.agent_pos = self.get_legal_pos()

    def generate_wall(self):
        """
        生成墙壁
        :return:
        """
        self.grid[0][:] = ELEMENTS.obstacle
        self.grid[self.height - 1][:] = ELEMENTS.obstacle
        self.grid[:, 0] = ELEMENTS.obstacle
        self.grid[:, self.width - 1] = ELEMENTS.obstacle

    def generate_rec(self, left_top, right_bottom):
        """
        通过两点生成矩形障碍物
        :param left_top:
        :param right_bottom:
        :return:
        """
        self.grid[max(0, left_top[X]):min(self.height - 1, right_bottom[X]),
        max(0, left_top[Y]):min(self.width - 1, right_bottom[Y])] = ELEMENTS.obstacle

    def emission(self, emission_step=4, dec=20):
        """
        将障碍物扩散，生成代价地图
        :param emission_step:
        :param dec:
        :return:
        """
        for x in range(self.height):
            for y in range(self.width):
                if self.cost_map[x][y] == 100:
                    self.emission_cell(x, y, emission_step, dec)

    def emission_cell(self, x, y, emission_step=4, dec=20):
        for i in range(1, emission_step + 1):
            left_top = self.limit_pos((x - i, y - i))
            right_bottom = self.limit_pos((x + i + 1, y + i + 1))
            slices = self.cost_map[left_top[X]:right_bottom[X], left_top[Y]:right_bottom[Y]]
            mask = slices < 100 - dec * i
            slices[mask] = 100 - dec * i

    def limit_pos(self, pos):
        """
        将点pos限制到地图中
        例如点(-2, 10)，其x坐标-2不合法，将其调整为(0, 10)
        :param pos:
        :return:
        """
        return min(max(0, pos[0]), self.height - 1), min(max(0, pos[1]), self.width - 1)

    def generate_obstacle_rand(self, o_num=30, low=7, high=11):
        """
        生成n个随机障碍物
        :param o_num:
        :param low:
        :param high:
        :return:
        """
        for i in range(o_num):
            left_top = self.get_rand_pos()
            right_bottom = (left_top[X] + np.random.randint(low, high), left_top[Y] + np.random.randint(low, high))
            self.generate_rec(left_top, right_bottom)
            self.obstacles.append((left_top, right_bottom))

    def get_rand_pos(self):
        """
        生成一个地图内的随机点
        :return:
        """
        return np.random.randint(0, self.height), np.random.randint(0, self.width)

    def get_legal_pos(self, view_size=5):
        """
        得到地图内的一个合法点(周围n*n均无障碍物)
        :return:
        """
        while True:
            pos = self.get_rand_pos()
            around = self.slices_map(pos, size=view_size, use_data='cost_map')
            mask = around > self.threshold
            if mask.sum() > 0:
                continue
            return pos

    def show_cost_map(self):
        plt.figure(1)
        plt.clf()
        show_cost_map(self.cost_map)
        plt.colorbar()

    def show_grid(self):
        plt.figure(2)
        plt.clf()
        show_grid(self.grid)

    def slices_map(self, pos, use_data="grid", size=3, grid_map=None):
        """
        以点pos为中心，切片一个(size, size)的地图
        :param grid_map:
        :param pos:
        :param use_data: cost_map or grid
        :param size: 只能为单数
        :return:
        """
        # assert size % 2 == 1
        if grid_map is not None:
            data = grid_map
        elif use_data == "cost_map":
            data = self.cost_map
        elif use_data == "grid":
            data = self.grid
        else:
            return None
        slices = np.zeros((size, size))
        x, y = pos[0], pos[1]
        x, y = x - int(size / 2), y - int(size / 2)
        for slices_x, _x in enumerate(range(x, x + size)):
            for slices_y, _y in enumerate(range(y, y + size)):
                if 0 <= _x < self.height and 0 <= _y < self.width:
                    slices[slices_x][slices_y] = data[_x][_y].copy()

        return slices

    def get_obstacles_points(self):
        """
        获取所有的障碍点
        :return:
        """

        return np.where(self.grid == 1)
