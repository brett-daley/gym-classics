from abc import ABCMeta, abstractmethod

import numpy as np

import gym_classics


if gym_classics._backend == 'gym':
     from gym import Env
     from gym.spaces import Discrete
elif gym_classics._backend == 'gymnasium':
    from gymnasium import Env
    from gymnasium.spaces import Discrete


class BaseEnv(Env, metaclass=ABCMeta):
    """Abstract base class for shared functionality between all environments."""

    def __init__(self, starts, n_actions, reachable_states=None):
        self._starts = tuple(starts)
        self.action_space = Discrete(n_actions)
        self.np_random = None  # Initialized by calling reset()

        self.state = None
        self._transition_cache = {}

        if reachable_states is None:
            # Get reachable states by searching through the state space
            self._reachable_states = set()
            for s in self._starts:
                self._search(s, self._reachable_states)
            self._reachable_states = frozenset(self._reachable_states)
        else:
            # Use the provided reachable states
            self._reachable_states = frozenset(reachable_states)

        # Make look-up tables for quick state-to-integer conversion and vice-versa
        self._encoder = {}
        self._decoder = {}
        i = 0
        for state in self._reachable_states:
            self._encoder[state] = i
            self._decoder[i] = state
            i += 1
        self.observation_space = Discrete(i)

    def _search(self, state, visited):
        """A recursive depth-first search that adds all reachable states to the visited set."""
        visited.add(state)
        for a in self.actions():
            for transition in self._generate_transitions(state, a):
                next_state, _, done, prob = transition
                if prob > 0.0:
                    if not done and next_state not in visited:
                        self._search(next_state, visited)

    def reset(self, seed=None):
        if self.np_random is None and seed is None:
            seed = np.random.default_rng().integers(2**32)

        if seed is not None:
            self.action_space.seed(seed)
            self.np_random = np.random.default_rng(seed)

        i = self.np_random.choice(len(self._starts))
        self.state = self._starts[i]
        return self.encode(self.state), {}

    def step(self, action):
        assert self.action_space.contains(action)
        state = self.state
        elements = self._sample_random_elements(state, action)
        next_state, reward, done, _ = self._deterministic_step(state, action, *elements)
        self.state = next_state
        return self.encode(next_state), reward, done, False, {}

    def _sample_random_elements(self, state, action):
        """Samples values for random elements (if any) that influence the environment
        transition from the current state-action pair (S, A).

        If the environment is deterministic, no need to override this method.
        """
        return ()

    def _deterministic_step(self, state, action, *random_elements):
        """An environment step that is deterministic conditioned on the given values
        of the random variables (if there are any).

        Do not override.
        """
        next_state, prob = self._next_state(state, action, *random_elements)
        reward = self._reward(state, action, next_state)
        done = self._done(state, action, next_state)
        if done:
            next_state = state
        return next_state, reward, done, prob

    @abstractmethod
    def _next_state(self, state, action, *random_elements):
        """Returns the next state S' induced by the state-action pair (S, A), which must
        be deterministic conditioned on the values of any random_elements. Also returns
        the probability that this particular transition occurred."""
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

    def encode(self, state):
        """Converts a raw state into a unique integer."""
        return self._encoder[state]

    def decode(self, i):
        """Reverts an encoded integer back to its raw state."""
        return self._decoder[i]

    def is_reachable(self, state):
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

        for ns, r, d, p in self._generate_transitions(self.decode(state), action):
            ns = self.encode(ns)
            dones[ns] = float(d)
            rewards[ns] = r
            probabilities[ns] += p

        assert (probabilities >= 0.0).all(), "transition probabilities must be nonnegative"
        assert abs(probabilities.sum() - 1.0) <= 0.01, "transition probabilities must sum to 1"

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
