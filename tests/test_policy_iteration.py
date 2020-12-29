import unittest

import gym

from gym_classics.dynamic_programming import policy_iteration
from gym_classics.envs.abstract.gridworld import Gridworld
from gym_classics.envs.jacks_car_rental import JacksCarRental
from tests.test_value_iteration import print_gridworld


class TestPolicyIteration(unittest.TestCase):
    def test_classic_gridworld(self):
        self._run_test('ClassicGridworld-v0', discount=0.9)


    def test_jacks_car_rental(self):
        self._run_test('JacksCarRental-v0', discount=0.9)


    def _run_test(self, env_id, discount):
        env = gym.make(env_id)
        assert isinstance(env, Gridworld) or isinstance(env, JacksCarRental)
        policy = policy_iteration(env, discount)

        print(env_id + ':')
        if isinstance(env, JacksCarRental):
            # Some extra bookkeeping allows us to treat the car rental as a gridworld
            env._dims = (21, 21)
            policy -= 5
        print_gridworld(env, policy, decimals=0, separator=' ', signed=False)
        print(flush=True)
