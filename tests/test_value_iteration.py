import unittest

import gym
import numpy as np

from gym_classics.dynamic_programming import value_iteration
from gym_classics.envs.abstract.gridworld import Gridworld
from gym_classics.envs.abstract.racetrack import Racetrack
from gym_classics.envs.jacks_car_rental import JacksCarRental
from gym_classics.utils import print_gridworld, print_racetrack


class TestValueIteration(unittest.TestCase):
    def test_5walk(self):
        self._run_test('5Walk-v0', discount=0.9)

    def test_19walk(self):
        self._run_test('19Walk-v0', discount=0.9)

    def test_classic_gridworld(self):
        self._run_test('ClassicGridworld-v0', discount=0.9)

    def test_cliff_walk(self):
        self._run_test('CliffWalk-v0', discount=0.9)

    def test_dyna_maze(self):
        self._run_test('DynaMaze-v0', discount=0.95)

    def test_four_rooms(self):
        self._run_test('FourRooms-v0', discount=0.95)

    def test_sparse_gridworld(self):
        self._run_test('SparseGridworld-v0', discount=0.9)

    def test_windy_gridworld(self):
        self._run_test('WindyGridworld-v0', discount=1.0)

    def test_windy_gridworld_kings(self):
        self._run_test('WindyGridworldKings-v0', discount=1.0)

    def test_windy_gridworld_kings_no_op(self):
        self._run_test('WindyGridworldKingsNoOp-v0', discount=1.0)

    def test_windy_gridworld_kings_stochastic(self):
        self._run_test('WindyGridworldKingsStochastic-v0', discount=1.0)


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


    def _run_test(self, env_id, discount):
        env = gym.make(env_id)
        V = value_iteration(env, discount)

        print(env_id + ':')
        if isinstance(env.unwrapped, Racetrack):
            # Since we can't simultaneously visualize position and velocity, print the
            # best value over all velocities in each position in the shape of the track
            print_racetrack(env, V)
        elif isinstance(env.unwrapped, Gridworld):
            # Print the values in the shape of the gridworld
            print_gridworld(env.unwrapped, V)
        elif isinstance(env, JacksCarRental):
            # Some extra bookkeeping allows us to treat the car rental as a gridworld
            env.dims = (21, 21)
            print_gridworld(env, V, decimals=0)
        else:
            # Just print a list of the encoded/decoded states and their values
            for s in env.states():
                print(s, ':', env.decode(s), ':', '{:.2f}'.format(V[s]))
        print(flush=True)
