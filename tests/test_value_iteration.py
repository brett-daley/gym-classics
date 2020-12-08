import unittest

import gym

import sutton_barto_gym
from sutton_barto_gym.value_iteration import value_iteration


class TestValueIteration(unittest.TestCase):
    def test_classic_gridworld(self):
        self._run_test('ClassicGridworld-v0', discount=0.9)

    def test_windy_gridworld(self):
        self._run_test('WindyGridworld-v0', discount=0.9)

    def _run_test(self, env_id, discount):
        env = gym.make(env_id)
        V = value_iteration(env, discount)
        print(env_id)
        print(env._decoder)
        print(V)
        print(flush=True)
