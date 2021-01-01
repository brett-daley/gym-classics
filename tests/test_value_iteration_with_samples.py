import unittest

import gym
import numpy as np

from gym_classics.dynamic_programming import value_iteration, value_iteration_with_samples


class TestValueIterationWithSamples(unittest.TestCase):
    def test_5walk(self):
        self._run_test('5Walk-v0', discount=0.9, deterministic=True)

    def test_19walk(self):
        self._run_test('19Walk-v0', discount=0.9, deterministic=True)

    # def test_classic_gridworld(self):
    #     self._run_test('ClassicGridworld-v0', discount=0.9)

    def test_cliff_walk(self):
        self._run_test('CliffWalk-v0', discount=0.9, deterministic=True)

    def test_dyna_maze(self):
        self._run_test('DynaMaze-v0', discount=0.95, deterministic=True)

    # def test_four_rooms(self):
    #     self._run_test('FourRooms-v0', discount=0.95)

    # def test_sparse_gridworld(self):
    #     self._run_test('SparseGridworld-v0', discount=0.9)

    def test_windy_gridworld(self):
        self._run_test('WindyGridworld-v0', discount=1.0, deterministic=True)

    def test_windy_gridworld_kings(self):
        self._run_test('WindyGridworldKings-v0', discount=1.0, deterministic=True)

    def test_windy_gridworld_kings_no_op(self):
        self._run_test('WindyGridworldKingsNoOp-v0', discount=1.0, deterministic=True)

    # def test_windy_gridworld_kings_stochastic(self):
    #     self._run_test('WindyGridworldKingsStochastic-v0', discount=1.0)


    # NOTE: the below tests are intentionally commented out because they're slow to run

    # def test_jacks_car_rental(self):
    #     self._run_test('JacksCarRental-v0', discount=0.9)

    # def test_jacks_car_rental_modified(self):
    #     self._run_test('JacksCarRentalModified-v0', discount=0.9)

    # def test_racetrack1(self):
    #     self._run_test('Racetrack1-v0', discount=0.9)

    # def test_racetrack2(self):
    #     self._run_test('Racetrack2-v0', discount=0.9)

    ###################################################################################


    def _run_test(self, env_id, discount, deterministic=False):
        np.random.seed(0)

        env = gym.make(env_id)
        env.seed(0)

        n = 1 if deterministic else 1000
        V1 = value_iteration_with_samples(env, discount, precision=1e-3, n=n)
        V2 = value_iteration(env, discount, precision=1e-3)

        if deterministic:
            self.assertTrue((V1 == V2).all())
        else:
            try:
                self.assertTrue(np.allclose(V1, V2, rtol=0.1, atol=0.1))
            except:
                print(V1)
                print(V2)
                print(V1 - V2)
                raise
