from gym.envs import register


register(
    id='ClassicGridworld-v0',
    entry_point='sutton_barto_gym.envs.classic_gridworld:ClassicGridworld'
)

register(
    id='WindyGridworld-v0',
    entry_point='sutton_barto_gym.envs.windy_gridworld:WindyGridworld'
)
