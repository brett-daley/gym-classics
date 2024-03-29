# Gym Classics
![pypi](https://img.shields.io/badge/pypi-1.0.0-blue)
[![license](https://img.shields.io/badge/license-GPL%20v3.0-blue)](./LICENSE)
![python](https://img.shields.io/badge/python-3.5%2B-green)

Gym Classics is a collection of well-known discrete MDPs from the reinforcement learning
literature implemented as OpenAI Gym environments.
API support for dynamic programming is also provided.

The environments include tasks across a range of difficulties, from small random walks
and gridworlds to challenging domains like racetracks and Jack's Car Rental.
These can be used as benchmarks for comparing the performance of various agents, or to
test and debug new learning methods.

### Contents

1. [Installation](#installation)

1. [API Overview](#api-overview)

1. [Example: Reinforcement Learning](#example-reinforcement-learning)

1. [Example: Dynamic Programming](#example-dynamic-programming)

1. [Environments Glossary](#environments-glossary)

1. [Citing and References](#citing-and-references)

## Installation

Prerequisites:
- Python 3.5+
- `gym==0.26.2` or `gymnasium`
- `numpy`
- `scipy` (for `JacksCarRental` and `JacksCarRentalModified` only)

### Option 1: `pip`

```
pip install gym-classics
```

### Option 2: `setuptools`

```
git clone https://github.com/brett-daley/gym-classics.git
cd gym-classics
python setup.py install
```

---

## API Overview

The basic API is identical to that of [OpenAI Gym](https://www.gymlibrary.dev/) (as of `0.26.2`) and [Gymnasium](https://gymnasium.farama.org/).
The environments must be explictly registered for `gym.make` by
importing the `gym_classics` package in your Python script and then calling `gym_classics.register('gym')` or `gym_classics.register('gymnasium')`, depending on which library you want to use as the backend.
Note that registration cannot be changed after calling `register`, and mixing `gym` and `gymnasium` environments in a single script is not possible.

A minimal working example:

```python
import gym                    # or `import gymnasium as gym`
import gym_classics
gym_classics.register('gym')  # or `gym_classics.register('gymnasium')`

env = gym.make('ClassicGridworld-v0')
state, _ = env.reset(seed=0)

for t in range(1, 100 + 1):
    action = env.action_space.sample()  # Select a random action
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    print("t={}, state={}, action={}, reward={}, next_state={}, done={}".format(
        t, state, action, reward, next_state, done))
    if done:
        next_state, _ = env.reset()
    state = next_state

env.close()  # Optional, not currently implemented by any environments
```

Gym Classics also implements methods for querying the model of the environment.
The full interface of a Gym Classics environment therefore looks like this:

```yaml
class Env:
    # Standard Gym API:
    - step(self, action)
    - reset(self, seed=None, options=None)
    - render(self)  # *currently not implemented by all environments*
    - close(self)

    # Extended Gym Classics API:
    - states(self)                # returns a generator over all feasible states
    - actions(self)               # returns a generator over all feasible actions
    - model(self, state, action)  # returns all transitions from the given state-action pair
```

The usage of `states`, `actions`, and `model` are discussed in
[Example: Dynamic Programming](#example-dynamic-programming).

State and action spaces for all environments are type `gym.spaces.Discrete`.
The size of these spaces can be queried as usual:
`env.observation_space.n` and `env.action_space.n`.
This means that states and actions are represented as unique integers, which is useful
for advanced `numpy` indexing.
Note that states and actions are enumerated in an arbitrary but consistent order for each environment.

> **Tip:** Gym Classics environments also implement methods called `encode` and `decode` which convert states between their integer and human-interpretable forms.
> These should never be used by the agent, but can be useful for displaying results or debugging.
> See the abstract [BaseEnv](gym_classics/envs/abstract/base_env.py) class for implementation details.

## Example: Reinforcement Learning

Let's test the classic Q-Learning algorithm [[4]](#references) on `ClassicGridworld-v0`.

```python
import gym
import gym_classics
gym_classics.register('gym')
import numpy as np

# Hyperparameters for Q-Learning
discount = 0.9
epsilon = 0.5
learning_rate = 0.025

# Instantiate the environment
env = gym.make('ClassicGridworld-v0')
state, _ = env.reset(seed=0)

# Our Q-function is a numpy array
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Loop for 500k timesteps
for _ in range(500000):
    # Select action from ε-greedy policy
    if env.np_random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])

    # Step the environment
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    # Q-Learning update:
    # Q(s,a) <-- Q(s,a) + α * (r + γ max_a' Q(s',a') - Q(s,a))
    td_error = reward - Q[state, action]
    if not done:
        td_error += discount * np.max(Q[next_state])
    Q[state, action] += learning_rate * td_error

    # Reset the environment if we're done
    if done:
        next_state, _ = env.reset()
    state = next_state

# Now let's see what the value function looks like after training:
V = np.max(Q, axis=1)
print(V)
```

Output:

```
[ 0.56303004  0.72570493  0.56160538  0.48701053 -1.          0.44497334
  0.2242687   0.63966295  0.84551377  0.42840196  1.        ]
```

These values seem reasonable, but in the next section, we will certify their correctness
by using dynamic programming.

## Example: Dynamic Programming

Gym Classics extends the OpenAI Gym API by providing a lean interface for dynamic
programming.
Generators are provided for the state and action spaces, enabling sweeps over the
state-action pairs:

```python
print(sorted(env.states()))
print(sorted(env.actions()))
```

Output:

```
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
[0, 1, 2, 3]
```

We can therefore see that `ClassicGridworld-v0` has 11 states and 4 actions.
Since the state/action generators always return elements in the same arbitrary order,
it is recommended to sort or shuffle them as needed.

It is also possible to poll the environment model at an arbitrary state-action pair.
Let's inspect the model at state 0 and action 1:

```python
next_states, rewards, dones, probabilities = env.model(state=0, action=1)
print(next_states)
print(rewards)
print(dones)
print(probabilities)
```

Output:

```
[ 0  3 10]
[0. 0. 0.]
[0. 0. 0.]
[0.8 0.1 0.1]
```

Each of the 4 return values are `numpy` arrays that represent the possible transitions.
In this case, there are 3 transitions from state 0 after taking action 1:

1. Go to state 0, yield +0 reward, do not terminate episode. Probability: 80%.
1. Go to state 3, yield +0 reward, do not terminate episode. Probability: 10%.
1. Go to state 10, yield +0 reward, do not terminate episode. Probability: 10%.

Note that these `numpy` arrays allow us to perform a value backup in a neat one-line
solution using advanced indexing!

```python
import numpy as np
V = np.zeros(env.observation_space.n)
V[0] = np.sum(probabilities * (rewards + discount * (1.0 - dones) * V[next_states]))
```

In practice, only advanced users will need to conduct backups manually like this.
Value Iteration and other dynamic programming methods are already implemented in
[dynamic_programming.py](gym_classics/dynamic_programming.py).
Let's use Value Iteration to check that our Q-Learning implemention from
[Example: Reinforcement Learning](#example-reinforcement-learning) is correct:

```python
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
```

Output:

```
[ 0.56631445  0.74438015  0.57185903  0.49068396 -1.          0.47547113
  0.27729584  0.64496924  0.84776628  0.43084446  1.        ]

RMS error: 0.01967779454940685
Maximum absolute difference: 0.053027139347779195
```

Both error metrics are very close to zero;
we can conclude that our Q-Learning implementation
is working!

## Environments Glossary

| # | Env ID | Description |
| :-: | :-: | --- |
| 1 | `5Walk-v0` | A 5-state deterministic linear walk. Ideal for implementing random walk experiments.<br><br>**reference:** [[3]](#references) (page 125).<br><br>**state:** Discrete position {0, ..., 4} on the number line.<br><br>**actions:** Move left/right.<br><br>**rewards:** +1 for moving right in the extreme right state.<br><br>**termination:** Moving right in the extreme right state or moving left in the extreme left state. |
| 2 | `19Walk-v0` | Same as `5Walk` but with 19 states and an additional -1 reward for moving left in the extreme left state.<br><br>**reference:** [[3]](#references) (page 145). |
| 3 | `ClassicGridworld-v0` | A 4x3 pedagogical gridworld. The agent starts in the bottom-left cell. Actions are noisy; with a 10% chance each, a move action may be rotated by 90 degrees clockwise or counter-clockwise (the "80-10-10 rule"). Cell (1, 1) is blocked and cannot be occupied by the agent.<br><br>**reference:** [[1]](#references) (page 646).<br><br>**state**: Grid location.<br><br>**actions**: Move up/right/down/left.<br><br>**rewards**: +1 for taking any action in cell (3, 2). -1 for taking any action in cell (3, 1). *NOTE:* The original version adds a -0.04 penalty to all other transitions, but this implementation does not.<br><br>**termination**: Earning a nonzero reward. |
| 4 | `CliffWalk-v0` | The Cliff Walking task, a 12x4 gridworld often used to contrast Sarsa with Q-Learning. The agent begins in the bottom-left cell and must navigate to the goal (bottom-right cell) without entering the region along the bottom ("The Cliff").<br><br>**reference:** [[3]](#references) (page 132, example 6.6).<br><br>**state**: Grid location.<br><br>**actions**: Move up/right/down/left.<br><br>**rewards**: -100 for entering The Cliff. -1 for all other transitions.<br><br>**termination**: Entering The Cliff or reaching the goal. |
| 5 | `DynaMaze-v0` | A 9x6 deterministic gridworld with barriers to make navigation more challenging. The agent starts in cell (0, 3); the goal is the top-right cell.<br><br>**reference:** [[3]](#references) (page 164, example 8.1).<br><br>**state**: Grid location.<br><br>**actions**: Move up/right/down/left.<br><br>**rewards**: +1 for episode termination.<br><br>**termination**: Reaching the goal. |
| 6 | `FourRooms-v0` | An 11x11 gridworld segmented into four rooms. The agent begins in the bottom-left cell; the goal is in the top-right cell. Actions are noisy; instead of the original transition probabilities, this implementation uses the 80-10-10 rule from `ClassicGridworld`.<br><br>**reference:** [[2]](#references) (page 192).<br><br>**state**: Grid location.<br><br>**actions**: Move up/right/down/left.<br><br>**rewards**: +1 for episode termination.<br><br>**termination**: Taking any action in the goal. |
| 7 | `JacksCarRental-v0` | A challenging management problem where a rental company must balance the number of cars between two parking lots to maximize its profit. On each timestep, Poisson-distributed numbers of requests and returns come into each lot. (The lots have different statistics.) The agent may then move up to 5 cars between the lots for a proportional fee. The lots can never have more than 20 cars each, and a lot earns money for a request only if it has a car available.<br><br>**reference:** [[3]](#references) (page 81, example 4.2).<br><br>**state:** The number of cars at both lots.<br><br>**actions:** Move a number of cars {-5, ..., 5} for a total of 9 actions. Positive numbers represent moving cars from lot 1 to lot 2; negative numbers represent moving cars from lot 2 to lot 1.<br><br>**rewards:** +10 for each satisfied rental request. -2 for each car moved.<br><br>**termination:** 100 timesteps elapse. |
| 8 | `JacksCarRentalModified-v0` | Same as `JacksCarRental` but with two modifications to the reward function. On each timestep:<br><br>1. One of Jack's employees can move a car from lot 1 to 2 for free.<br><br>2. Overnight parking incurs -4 reward per lot with more than 10 cars.<br><br>**reference:** [[3]](#references) (page 82, exercise 4.7). |
| 9 | `Racetrack1-v0` | A gridworld-type racetrack where a racecar must traverse a right turn and reach the finish line as quickly as possible. The agent begins at a random location on the starting line and can only directly control the velocity of the racecar (not its position). Each velocity component can never be negative nor greater than 4. If the car goes out of bounds, it is reset to a random location on the starting line without terminating the episode. *NOTE:* While the original version forbids both velocity components from being zero simultaneously, no such restriction is enforced in this implementation.<br><br>**reference:** [[3]](#references) (page 112, figure 5.5, left).<br><br>**state:** Racecar position and velocity.<br><br>**actions:** Changes to the racecar's *velocity* (not position) vector, where the x- and y- components can be independently modified by {-1, 0, +1} on each timestep. This gives a total of 9 actions.<br><br>**rewards:** -1 on all transitions unless the finish line is reached.<br><br>**termination:** Reaching the finish line. |
| 10 | `Racetrack2-v0` | Same as `Racetrack1` but with a different track layout.<br><br>**reference:** [[3]](#references) (page 112, figure 5.5, right). |
| 11 | `SparseGridworld-v0` | A 10x8 featureless gridworld. The agent starts in cell (1, 3) and the goal is at cell (6, 3). To make it more challenging, the same 80-10-10 transition probabilities from `ClassicGridworld` are used. Great for testing various forms of credit assignment in the presence of noise.<br><br>**reference:** [[3]](#references) (page 147, figure 7.4).<br><br>**states:** Grid location.<br><br>**actions:** Move up/right/down/left.<br><br>**rewards:** +1 for episode termination.<br><br>**termination:** Reaching the goal. |
| 12 | `WindyGridworld-v0` | A 10x7 deterministic gridworld where some columns are affected by an upward wind. The agent starts in cell (0, 3) and the goal is at cell (7, 3). If an agent executes an action from a cell with wind, the resulting position is given by the vector sum of the action's effect and the wind.<br><br>**reference:** [[3]](#references) (page 130, example 6.5).<br><br>**state:** Grid location.<br><br>**actions:** Move up/right/down/left.<br><br>**rewards:** -1 for all transitions unless the episode terminates.<br><br>**termination:** Reaching the goal. |
| 13 | `WindyGridworldKings-v0` | Same as `WindyGridworld` but with diagonal "King's" moves permitted.<br><br>**reference:** [[3]](#references) (page 131, exercise 6.9).<br><br>**actions:** Move in the 4 cardinal directions and 4 intermediate directions. |
| 14 | `WindyGridworldKingsNoOp-v0` | Same as `WindyGridworldKings` but with an extra "no-op" (do nothing) action.<br><br>**reference:** [[3]](#references) (page 131, exercise 6.9).<br><br>**actions:** Move in the 8 cardinal/intermediate directions or take a no-op action. |
| 15 | `WindyGridworldKingsStochastic-v0` | Same as `WindyGridworldKings` but windy cells exhibit stochastic behavior: -1, +0, or +1 wind strength with probability 1/3 each.<br><br>**reference:** [[3]](#references) (page 131, exercise 6.10). |


---

## Citing and References

You can cite this repository in published work using the following bibtex:

```
@misc{daley2021gym,
  author={Daley, Brett},
  title={Gym Classics},
  year={2021},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/brett-daley/gym-classics}},
}
```

### References

1. [Russell & Norvig. Artificial Intelligence: A Modern Approach. 2009, 3rd Ed.](https://cs.calvin.edu/courses/cs/344/kvlinden/resources/AIMA-3rd-edition.pdf)

1. [Sutton, Precup, & Singh. Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning. 1999.](https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf)

1. [Sutton & Barto. Reinforcement Learning: An Introduction. 2018, 2nd Ed.](http://incompleteideas.net/book/RLbook2020.pdf)

1. [Watkins. Learning from Delayed Rewards. 1989.](http://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf)
