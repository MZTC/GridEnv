# *_* coding : UTF-8 *_*
# @institution: XJU
# @author: MZT
# @time: 2023/5/8  22:24
# @file name: gird_demo.py
import grid_env
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('grid_env', width=60)
obs, info = env.reset()
while True:
    plt.figure(1)
    for i in range(len(obs)):
        plt.subplot(1, 3, i + 1)
        plt.imshow(obs[i], cmap="gray")
    plt.draw()
    plt.show(block=False)
    plt.pause(0.0001)
    action = env.action_space.sample()
    obs, r, ter, tur, info = env.step(action)
    print(info)
    env.render()
    if ter | tur:
        obs, info = env.reset()
