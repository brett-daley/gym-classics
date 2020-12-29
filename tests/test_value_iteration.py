import unittest

import gym
import numpy as np

from gym_classics.envs.abstract.gridworld import Gridworld
from gym_classics.envs.abstract.racetrack import Racetrack
from gym_classics.envs.jacks_car_rental import JacksCarRental
from gym_classics.value_iteration import value_iteration


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

    # def test_racetrack1(self):
    #     self._run_test('Racetrack1-v0', discount=0.9)

    # def test_racetrack2(self):
    #     self._run_test('Racetrack2-v0', discount=0.9)

    ###################################################################################


    def _run_test(self, env_id, discount):
        env = gym.make(env_id)
        V = value_iteration(env, discount)

        print(env_id + ':')
        if isinstance(env, Racetrack):
            # Since we can't simultaneously visualize position and velocity, print the
            # best value over all velocities in each position in the shape of the track
            self._print_racetrack(env, V)
        elif isinstance(env, Gridworld):
            # Print the values in the shape of the gridworld
            self._print_gridworld(env, V)
        elif isinstance(env, JacksCarRental):
            # Some extra bookkeeping allows us to treat the car rental as a gridworld
            env._dims = (21, 21)
            self._print_gridworld(env, V, decimals=0)
        else:
            # Just print a list of the encoded/decoded states and their values
            for s in env.states():
                print(s, ':', env._decode(s), ':', '{:.2f}'.format(V[s]))
        print(flush=True)

    def _print_racetrack(self, env, V):
        grid_values = np.zeros(env._dims)

        reachable_positions = [pos for (pos, _) in env._reachable_states]
        velocities = [(x, y) for x in range(env._max_velocity + 1)
                             for y in range(env._max_velocity + 1)]

        # Helper function to get reachable velocity pairings in this position
        def valid_states_in_position(pos):
            states = [(pos, vel) for vel in velocities]
            return np.asarray([env._encode(s) for s in states if s in env._reachable_states])

        # Set each cell to be the maximum value over the velocities
        for x in range(env._dims[0]):
            for y in range(env._dims[1]):
                pos = (x, y)
                if pos in reachable_positions:
                    states = valid_states_in_position(pos)
                    grid_values[x, y] = np.max(V[states])
                else:
                    grid_values[x, y] = np.nan

        # First get the string length of the longest number
        formatter = lambda v: '{:+.2f}'.format(v)
        maxlen = max([len(formatter(v)) for v in grid_values.flatten()])

        # Now we can actually print the values
        for y in reversed(range(env._dims[1])):
            for x in range(env._dims[0]):
                v = grid_values[x,y]
                if not np.isnan(v):
                    print(formatter(v).rjust(maxlen), end=' ')
                else:
                    print(' ' * maxlen, end=' ')
            print()

    def _print_gridworld(self, env, V, decimals=2):
        # First get the string length of the longest number
        formatter = lambda v: ('{:+.' + str(decimals) + 'f}').format(v)
        maxlen = max([len(formatter(v)) for v in V])

        # Now we can actually print the values
        for y in reversed(range(env._dims[1])):
            for x in range(env._dims[0]):
                state = (x, y)
                if env._is_reachable(state):
                    s = env._encode(state)
                    print(formatter(V[s]).rjust(maxlen), end=' ' * 2)
                else:
                    print(' ' * maxlen, end=' ' * 2)
            print()
