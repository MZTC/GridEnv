# *_* coding : UTF-8 *_*
# @team: XJU
# @author: MZT
# @time: 2023/3/30  18:16
# @file name: env.py
from typing import SupportsFloat, Any, Tuple, Dict

import numpy as np
from gymnasium.core import Env, ActType, ObsType
from gymnasium.spaces import Discrete, Box
from matplotlib import pyplot as plt

from grid_env.core.action import Action
from grid_env.core.grid import Grid
from grid_env.core.constants import ELEMENTS


class GridEnv(Env):

    def __init__(self, width=150, height=150,
                 threshold=70, obstacle_num=16,
                 emission_step=4, emission_dec=20,
                 view_size=15, mode='grid', max_step=None):
        self.width = width
        self.height = height
        self.threshold = threshold
        self.obstacle_num = obstacle_num
        self.emission_step = emission_step
        self.emission_dec = emission_dec
        self.view_size = view_size
        self.mode = mode

        if max_step is None:
            self.max_step = max(width, height) * 2
        else:
            self.max_step = max_step

        # 初始化栅格地图
        self.grid = Grid(width=self.width, height=self.height,
                         threshold=self.threshold, obstacle_num=self.obstacle_num,
                         emission_step=self.emission_step, emission_dec=self.emission_dec)

        self.stepc = 0

    @property
    def action_space(self):
        action_num = len(Action)
        return Discrete(action_num)

    @property
    def observation_space(self):
        return Box(-10, 10, (3, self.view_size * 2, self.view_size * 2))

    def _ge_obs(self):

        obs_layer = self.grid.grid.copy()
        agent_layer = np.zeros_like(obs_layer)
        goal_layer = np.zeros_like(obs_layer)
        agent_layer[self.grid.agent_pos[0]][self.grid.agent_pos[1]] = 1
        goal_layer[self.grid.goal[0]][self.grid.goal[1]] = 1

        obs_layer_s = self.grid.slices_map(self.grid.agent_pos, use_data=self.mode,
                                           size=self.view_size * 2, grid_map=obs_layer)

        agent_layer_s = self.grid.slices_map(self.grid.agent_pos, use_data=self.mode,
                                             size=self.view_size * 2, grid_map=agent_layer)

        goal_layer_s = self.grid.slices_map(self.grid.agent_pos, use_data=self.mode,
                                            size=self.view_size * 2, grid_map=goal_layer)

        return np.array([obs_layer_s, agent_layer_s, goal_layer_s])

    def reset(self):
        self.grid.reset_map()
        self.stepc = 0

        return self._ge_obs(), {"agent_pos": self.grid.agent_pos, "goal": self.grid.goal}

    def render(self):
        plt.figure("render")
        plt.clf()
        plt.cla()
        obstacles = self.grid.get_obstacles_points()
        plt.scatter(obstacles[1], obstacles[0], marker="s", c='black')
        plt.scatter(self.grid.agent_pos[1], self.grid.agent_pos[0], marker="s", c='red')
        plt.scatter(self.grid.goal[1], self.grid.goal[0], marker="s", c='green')
        plt.draw()
        plt.show(block=False)
        plt.pause(0.001)

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:

        action = int(action)
        action = Action[action]
        self.stepc += 1
        info = {
            "collision": False,
            "reward": 0,
            "succ": False,
            "stepc": self.stepc
        }

        reward = -0.01
        ter = False
        tur = False

        agent_x, agent_y = self.grid.agent_pos + np.array(action)

        next_agent_pos = np.array([agent_x, agent_y])
        agent_pos = np.array(self.grid.agent_pos)
        goal = np.array(self.grid.goal)

        if self.grid.grid[agent_x][agent_y] == ELEMENTS.obstacle:
            reward -= 1
            info['collision'] = True
        else:
            agent_pos = self.grid.agent_pos = next_agent_pos

        if (agent_pos == goal).all():
            reward += 100
            ter = True
            info["succ"] = True

        if self.stepc > self.max_step:
            tur = True

        info["reward"] = reward
        return self._ge_obs(), reward, ter, tur, info
