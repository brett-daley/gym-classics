import unittest

import gym

import gym_classics
gym_classics.register('gym')
from gym_classics.dynamic_programming import policy_iteration
from gym_classics.envs.abstract.gridworld import Gridworld
from gym_classics.envs.jacks_car_rental import JacksCarRental
from gym_classics.utils import print_gridworld


class TestPolicyIteration(unittest.TestCase):
    def test_classic_gridworld(self):
        self._run_test('ClassicGridworld-v0', discount=0.9)


    def test_jacks_car_rental(self):
        self._run_test('JacksCarRental-v0', discount=0.9)


    def test_jacks_car_rental_modified(self):
        self._run_test('JacksCarRentalModified-v0', discount=0.9)


    def _run_test(self, env_id, discount):
        env = gym.make(env_id)
        assert isinstance(env.unwrapped, (Gridworld, JacksCarRental))
        policy = policy_iteration(env, discount)

        print(env_id + ':')
        kwargs = dict(decimals=0, separator=' ', signed=False)

        if isinstance(env.unwrapped, JacksCarRental):
            env.dims = (21, 21)
            policy -= 5
            kwargs['transpose'] = True

        print_gridworld(env, policy, **kwargs)
        print(flush=True)
