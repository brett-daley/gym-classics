import gym
import gym_classics
gym_classics.register('gym')
from gym_classics.dynamic_programming import value_iteration
import numpy as np

# Instantiate the environment
env = gym.make('ClassicGridworld-v0')

# Compute the near-optimal values with Value Iteration
V_star = value_iteration(env, discount=0.9, precision=1e-9)
print(V_star, end='\n\n')

# Our Q-Learning values from earlier:
V = [0.56303004, 0.72570493, 0.56160538, 0.48701053, -1., 0.44497334,
     0.2242687,  0.63966295, 0.84551377, 0.42840196, 1.]

# Root Mean Square error:
rms_error = np.sqrt(np.mean(np.square(V - V_star)))
print("RMS error:", rms_error)

# Maximum absolute difference:
max_abs_diff = np.max(np.abs(V - V_star))
print("Maximum absolute difference:", max_abs_diff)
