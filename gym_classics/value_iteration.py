import argparse

import gym
import numpy as np


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
                next_states, rewards, dones, probs = env.model(s, a)
                bootstraps = (1.0 - dones) * V[next_states]
                q = np.sum(probs * (rewards + discount * bootstraps))
                q_values.append(q)
            V[s] = max(q_values)
            delta = max(delta, abs(V[s] - V_old))

        if delta <= max_delta:
            break

    return V
