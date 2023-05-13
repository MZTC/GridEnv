# GridEnv
这是一个简单的二维栅格地图导航环境
![osb](/image/obs.png)
![render](/image/render.png)

# 快速使用

```python
import grid_env
import gymnasium as gym

env = gym.make('grid_env', width=60)
obs, info = env.reset()
```

