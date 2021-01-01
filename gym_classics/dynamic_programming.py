import argparse

import gym
import numpy as np


def value_iteration(env, discount, precision=1e-3):
    assert 0.0 <= discount <= 1.0
    assert precision > 0.0
    V = np.zeros(env.observation_space.n, dtype=np.float64)

    while True:
        V_old = V.copy()

        for s in env.states():
            Q_values = [backup(env, discount, V, s, a) for a in env.actions()]
            V[s] = max(Q_values)

        if np.abs(V - V_old).max() <= precision:
            return V


def value_iteration_with_samples(env, discount, precision=1e-3, n=100):
    model = env.model
    env.model = SampleBackup(env, n)
    V = value_iteration(env, discount, precision)
    env.model = model
    return V


def policy_iteration(env, discount, precision=1e-3):
    assert 0.0 <= discount <= 1.0
    assert precision > 0.0

    # For the sake of determinism, we start with the policy that always chooses action 0
    policy = np.zeros(env.observation_space.n, dtype=np.int32)

    while True:
        V_policy = policy_evaluation(env, discount, policy, precision)
        policy, stable = policy_improvement(env, discount, policy, V_policy, precision)
        if stable:
            return policy


def policy_evaluation(env, discount, policy, precision=1e-3):
    assert 0.0 <= discount <= 1.0
    assert precision > 0.0
    V = np.zeros(policy.shape, dtype=np.float64)

    while True:
        V_old = V.copy()

        for s in env.states():
            V[s] = backup(env, discount, V, s, policy[s])

        if np.abs(V - V_old).max() <= precision:
            return V


####################
# Helper functions #
####################


def policy_improvement(env, discount, policy, V_policy, precision=1e-3):
    policy_old = policy.copy()
    V_old = V_policy.copy()

    for s in env.states():
        Q_values = [backup(env, discount, V_policy, s, a) for a in env.actions()]
        policy[s] = np.argmax(Q_values)
        V_policy[s] = max(Q_values)

    stable = np.logical_or(
        policy == policy_old,
        np.abs(V_policy - V_old).max() <= precision,
    ).all()

    return policy, stable


def backup(env, discount, V, state, action):
    next_states, rewards, dones, probs = env.model(state, action)
    bootstraps = (1.0 - dones) * V[next_states]
    return np.sum(probs * (rewards + discount * bootstraps))


class SampleBackup:
    def __init__(self, env, n):
        self._env = env
        self._n = n
        self._cache = {}

    def __call__(self, state, action):
        sa_pair = (state, action)
        if sa_pair not in self._cache:
            self._add_to_cache(state, action)
        return self._cache[sa_pair]

    def _add_to_cache(self, state, action):
        env = self._env

        if hasattr(env, '_t'):
            t = env._t
        else:
            t = 0

        n = env.observation_space.n
        next_states = np.arange(n)
        dones = np.zeros(n)
        rewards = np.zeros(n)
        probabilities = np.zeros(n)

        for _ in range(self._n):
            # Reset environment state each time
            env._state = env._decode(state)

            # Sample an outcome from this state-action pair
            ns, r, d, _ = env.step(action)
            env._t = t

            dones[ns] = float(d)
            rewards[ns] = r
            probabilities[ns] += 1.0

        probabilities /= probabilities.sum()
        np.nan_to_num(probabilities, copy=False)

        i = np.nonzero(probabilities)
        transition = (next_states[i], rewards[i], dones[i], probabilities[i])
        self._cache[(state, action)] = transition
