from gym.envs import register


register(
    id='ClassicGridworld-v0',
    entry_point='sutton_barto_gym.envs.classic_gridworld:ClassicGridworld'
)

register(
    id='DynaMaze-v0',
    entry_point='sutton_barto_gym.envs.dyna_maze:DynaMaze'
)

register(
    id='WindyGridworld-v0',
    entry_point='sutton_barto_gym.envs.windy_gridworld:WindyGridworld'
)

register(
    id='WindyGridworldKings-v0',
    entry_point='sutton_barto_gym.envs.windy_gridworld:WindyGridworldKings'
)

register(
    id='WindyGridworldKingsNoOp-v0',
    entry_point='sutton_barto_gym.envs.windy_gridworld:WindyGridworldKingsNoOp'
)

register(
    id='WindyGridworldKingsStochastic-v0',
    entry_point='sutton_barto_gym.envs.windy_gridworld:WindyGridworldKingsStochastic'
)
