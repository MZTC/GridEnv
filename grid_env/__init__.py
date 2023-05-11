# *_* coding : UTF-8 *_*
# @institution: XJU
# @author: MZT
# @time: 2023/5/8  22:29
# @file name: __init__.py
from gymnasium import register

register(
    'grid_env',
    entry_point="grid_env.core.env:GridEnv"
)

