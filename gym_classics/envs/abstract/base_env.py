from abc import ABCMeta, abstractmethod

import gym
from gym.spaces import Discrete
from gym.utils import seeding
import numpy as np


class BaseEnv(gym.Env, metaclass=ABCMeta):
    """Abstract base class for shared functionality between all environments."""

    def __init__(self, starts, n_actions):
        self._starts = tuple(starts)
        self.action_space = Discrete(n_actions)

        self._state = None
        self._transition_cache = {}

        # Get reachable states by searching through the state space
        self._reachable_states = set()
        for s in self._starts:
            self._search(s, self._reachable_states)
        self._reachable_states = frozenset(self._reachable_states)

        # Make look-up tables for quick state-to-integer conversion and vice-versa
        self._encoder = {}
        self._decoder = {}
        i = 0
        for state in self._reachable_states:
            self._encoder[state] = i
            self._decoder[i] = state
            i += 1
        self.observation_space = Discrete(i)

        # Initialize the np_random module (user can override the seed later if desired)
        self.seed()

    def _search(self, state, visited):
        """A recursive depth-first search that adds all reachable states to the visited set."""
        visited.add(state)
        for a in self.actions():
            for transition in self._generate_transitions(state, a):
                next_state, _, done, prob = transition
                if prob > 0.0:
                    if not done and next_state not in visited:
                        self._search(next_state, visited)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        return [seed]

    def reset(self):
        i = self.np_random.choice(len(self._starts))
        self._state = self._starts[i]
        return self._encode(self._state)

    def step(self, action):
        assert self.action_space.contains(action)
        next_state, reward, done = self._sample_step(self._state, action)
        self._state = next_state
        return self._encode(next_state), reward, done, {}

    def _sample_step(self, state, action):
        """Samples an environment transition from the current state-action pair (S, A).

        If the environment is deterministic, no need to override this method.
        """
        next_state, reward, done, _ = self._deterministic_step(state, action)
        return next_state, reward, done

    def _deterministic_step(self, state, action, *variables):
        """An environment step that is deterministic conditioned on the given values
        of the random variables (if there are any).

        Do not override.
        """
        next_state, prob = self._next_state(state, action, *variables)
        reward = self._reward(state, action, next_state)
        done = self._done(state, action, next_state)
        if done:
            next_state = state
        return next_state, reward, done, prob

    @abstractmethod
    def _next_state(self, state, action, **random_variables):
        """Returns a (possibly random) next state S' induced by the state-action pair (S, A)
        along with its probability of occurence."""
        raise NotImplementedError

    @abstractmethod
    def _reward(self, state, action, next_state):
        """Returns the reward yielded by this (S,A,S') outcome."""
        raise NotImplementedError

    @abstractmethod
    def _done(self, state, action, next_state):
        """Returns True if this (S,A,S') outcome should terminate, False otherwise."""
        raise NotImplementedError

    def states(self):
        """Returns a generator over all possible environment states."""
        return range(self.observation_space.n)

    def _encode(self, state):
        """Converts a raw state into a unique integer."""
        return self._encoder[state]

    def _decode(self, i):
        """Reverts an encoded integer back to its raw state."""
        return self._decoder[i]

    def _is_reachable(self, state):
        """Returns True if the state can be reached from at least one start location,
        False otherwise."""
        return state in self._reachable_states

    def actions(self):
        """Returns a generator over all possible agent actions."""
        return range(self.action_space.n)

    def model(self, state, action):
        """Returns the transitions from the given state-action pair."""
        sa_pair = (state, action)
        if sa_pair in self._transition_cache:
            return self._transition_cache[sa_pair]

        n = self.observation_space.n
        next_states = np.arange(n)
        dones = np.zeros(n)
        rewards = np.zeros(n)
        probabilities = np.zeros(n)

        for ns, r, d, p in self._generate_transitions(self._decode(state), action):
            ns = self._encode(ns)
            dones[ns] = float(d)
            rewards[ns] = r
            probabilities[ns] += p

        assert np.allclose(probabilities.sum(), 1.0), "transition probabilities must sum to 1"
        i = np.nonzero(probabilities)
        transition = (next_states[i], rewards[i], dones[i], probabilities[i])
        self._transition_cache[sa_pair] = transition
        return transition

    @abstractmethod
    def _generate_transitions(self, state, action):
        """Returns a generator over all transitions from this state-action pair.

        Should be overridden in the subclass.
        """
        raise NotImplementedError
