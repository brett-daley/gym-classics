# Gym Classics
![pypi](https://img.shields.io/badge/pypi-0.0.0-blue)
[![license](https://img.shields.io/badge/license-GPL%20v3.0-blue)](./LICENSE)
![python](https://img.shields.io/badge/python-3.5%2B-blue)

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

1. [Example: Reinforcement Learning](#reinforcement-learning)

1. [Example: Dynamic Programming](#dynamic-programming)

1. [Environments Glossary](#environments-glossary)

1. [References](#references)

### Citing

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

## Installation

Prerequisites:
- python 3.5+
- gym
- numpy
- scipy (`JacksCarRental-v0` and `JacksCarRentalModified-v0` only)

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

Once installed, the environments are automatically registered for `gym.make` by
importing the `gym_classics` package in your Python script.
The basic API is identical to that of OpenAI Gym.
A minimal working example:

```python
import gym
import gym_classics

env = gym.make('ClassicGridworld-v0')
env.reset()
for _ in range(100):
    env.step(env.action_space.sample())  # Take a random action
env.close()
```

Gym Classics also implements methods for querying a model of the environment.
The full interface of a Gym Classics environment therefore looks like this:

```yaml
class Env:
    # Standard Gym API:
    - step(self, action)
    - reset(self)
    - render(self, mode='human')  # *currently not implemented*
    - close(self)
    - seed(self, seed=None)

    # Extended Gym Classics API:
    - states(self)                # returns a generator over all feasible states
    - actions(self)               # returns a generator over all feasible actions
    - model(self, state, action)  # returns all transitions from the given state-action pair
```

The usage of `states`, `actions`, and `model` are discussed in
[Example: Dynamic Programming](#example:-dynamic-programming).

State and action spaces for all environments are type `gym.spaces.Discrete`.
The size of these spaces can be queried as usual:
`env.observation_space.n` and `env.action_space.n`.
This means that states and actions are represented as unique integers, which is useful
for advanced `numpy` indexing.
Note that states and actions are enumerated in an arbitrary order for each environment.

> **Tip:** Gym Classics environments also implement private methods called `_encode` and `_decode` which convert states between their integral and human-interpretable forms.
> These should never be used by the agent, but can be useful for displaying results or debugging.
> See the abstract [BaseEnv](gym_classics/envs/abstract/base_env.py) class for implementation details.

## Example: Reinforcement Learning

Let's test the classic Q-Learning algorithm cite{4} on `ClassicGridworld-v0`.

```python
import gym
import gym_classics
import numpy as np

# Hyperparameters for Q-Learning
discount = 0.9
epsilon = 0.5
learning_rate = 0.025

# Instantiate the environment
env = gym.make('ClassicGridworld-v0')
state = env.reset()

# Set seeds for reproducibility
np.random.seed(0)
env.seed(0)

# Our Q-function is a numpy array
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Loop for 500k timesteps
for _ in range(500000):
    # Select action from ε-greedy policy
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])

    # Step the environment
    next_state, reward, done, _ = env.step(action)

    # Q-Learning update:
    # Q(s,a) <-- Q(s,a) + α * (r + γ max_a' Q(s',a') - Q(s,a))
    target = reward - Q[state, action]
    if not done:
        target += discount * np.max(Q[next_state])
    Q[state, action] += learning_rate * target

    # Reset the environment if we're done
    state = env.reset() if done else next_state
```

Now let's see what the value function looks like after training:

```python
V = np.max(Q, axis=1)
print(V)
```

Output:

```
[ 0.5618515   0.75169693  1.          0.49147301  0.26363411 -1.
  0.58655406  0.51379727  0.86959422  0.43567445  0.64966203]
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
V = np.zeros(env.observation_space.n)
V[0] = np.sum(probabilities * (rewards + discount * (1.0 - dones) * V[next_states]))
```

In practice, only advanced users will need to conduct backups manually like this.
Value Iteration and other dynamic programming methods are already implemented in
[dynamic_programming.py](gym_classics/dynamic_programming.py).
Let's use Value Iteration to check that our Q-Learning implemention from
[Example: Reinforcement Learning](#example:-reinforcement-learning) is correct:

```python
# Compute the near-optimal values with Value Iteration
from gym_classics.dynamic_programming import value_iteration
V_star = value_iteration(env, discount=0.9, precision=1e-9)

# Our Q-Learning values from earlier:
V = [0.5618515,  0.75169693, 1.,         0.49147301, 0.26363411, -1.,
     0.58655406, 0.51379727, 0.86959422, 0.43567445, 0.64966203]

# Root Mean Square error
print("RMS error: {.f}".for np.sqrt(np.square(V - V_star).mean()))

# Maximum absolute difference
print("Max abs diff: {.f}", np.abs(V - V_star).max())
```

Output:

```
RMS error: 0.014976847878141084
Max abs diff: 0.03832613967716292
```

Both error metrics are very close to zero;
we can conclude that our Q-Learning implementation
is working!

## Environments Glossary

| Env ID | Description |
| :-: | --- |
GLOSSARY

---

## References

1. [Russell & Norvig. Artificial Intelligence: A Modern Approach. 2009, 3rd Ed.](https://cs.calvin.edu/courses/cs/344/kvlinden/resources/AIMA-3rd-edition.pdf)

1. [Sutton, Precup, & Singh. Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning. 1999.](https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf)

1. [Sutton & Barto. Reinforcement Learning: An Introduction. 2018, 2nd Ed.](http://incompleteideas.net/book/RLbook2020.pdf)

1. [Watkins. Learning from Delayed Rewards. 1989.](http://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf)
