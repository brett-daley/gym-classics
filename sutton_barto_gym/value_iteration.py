import argparse

import gym
import numpy as np

import sutton_barto_gym


def value_iteration(env, discount, max_delta=1e-3):
    assert 0.0 <= discount <= 1.0
    assert max_delta > 0.0
    V = np.zeros(shape=[env.observation_space.n])

    while True:
        delta = 0.0

        for s in env.states():
            q_values = []
            V_old = V[s]
            for a in env.actions():
                next_state, done, reward, prob = env.transitions(s, a)
                bootstrap = (1.0 - done) * V[next_state]
                q = np.sum(prob * (reward + discount * bootstrap))
                q_values.append(q)
            V[s] = max(q_values)
            delta = max(delta, abs(V[s] - V_old))

        if delta <= max_delta:
            break

    return V
