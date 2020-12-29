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

    stable = np.logical_or(
        policy == policy_old,
        np.abs(V_policy - V_old).max() <= precision,
    ).all()

    return policy, stable


def backup(env, discount, V, state, action):
    next_states, rewards, dones, probs = env.model(state, action)
    bootstraps = (1.0 - dones) * V[next_states]
    return np.sum(probs * (rewards + discount * bootstraps))
