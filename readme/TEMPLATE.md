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

PYTHON{examples/random_policy.py}

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

Let's test the classic Q-Learning algorithm cite{4} on `ClassicGridworld-v0`.

PYTHON{examples/q_learning.py}

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

PYTHON{examples/value_iteration.py}

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
GLOSSARY

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
