import gym
from gym_classics.dynamic_programming import value_iteration
import numpy as np

# Instantiate the environment
env = gym.make('ClassicGridworld-v0')
state = env.reset()

# Set seeds for reproducibility
np.random.seed(0)
env.seed(0)

# Compute the near-optimal values with Value Iteration
V_star = value_iteration(env, discount=0.9, precision=1e-9)

# Our Q-Learning values from earlier:
V = [0.5618515,  0.75169693, 1.,         0.49147301, 0.26363411, -1.,
     0.58655406, 0.51379727, 0.86959422, 0.43567445, 0.64966203]

# Root Mean Square error:
print("RMS error: {}".format(np.sqrt(np.square(V - V_star).mean())))

# Maximum absolute difference:
print("Max abs diff: {}".format(np.abs(V - V_star).max()))
