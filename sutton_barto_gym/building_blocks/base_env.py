import gym
import numpy as np


class BaseEnv(gym.Env):
    """Abstract base class for shared functionality between all environments."""

    def __init__(self):
        self.observation_space = None
        self.action_space = None
        self._transition_cache = {}

    def seed(self, seed=None):
        self.action_space.seed(seed)
        return [seed]

    def states(self):
        """Returns a generator over all possible environment states."""
        return range(self.observation_space.n)

    def actions(self):
        """Returns a generator over all possible agent actions."""
        return range(self.action_space.n)

    def transitions(self, state, action):
        """Returns the transitions from the given state-action pair."""
        sa_pair = (state, action)
        if sa_pair in self._transition_cache:
            return self._transition_cache[sa_pair]

        n = self.observation_space.n
        next_states = np.arange(n)
        dones = np.zeros(n)
        rewards = np.zeros(n)
        probabilities = np.zeros(n)

        for s, r, p, d in self._generate_transitions(state, action):
            dones[s] = float(d)
            rewards[s] = r
            probabilities[s] += p

        i = np.nonzero(probabilities)

        assert np.allclose(probabilities.sum(), 1.0)
        transition = (next_states[i], dones[i], rewards[i], probabilities[i])
        self._transition_cache[sa_pair] = transition
        return transition

    def _generate_transitions(self, state, action):
        """Returns a generator over all transitions from this state-action pair.

        Should be overridden in the subclass.
        """
        raise NotImplementedError
