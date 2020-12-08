import unittest

import gym

from sutton_barto_gym.envs.abstract.gridworld import Gridworld
from sutton_barto_gym.value_iteration import value_iteration


class TestValueIteration(unittest.TestCase):
    def test_classic_gridworld(self):
        self._run_test('ClassicGridworld-v0', discount=0.9)

    def test_dyna_maze(self):
        self._run_test('DynaMaze-v0', discount=0.95)

    def test_windy_gridworld(self):
        self._run_test('WindyGridworld-v0', discount=1.0)

    def test_windy_gridworld_kings(self):
        self._run_test('WindyGridworldKings-v0', discount=1.0)

    def test_windy_gridworld_kings_no_op(self):
        self._run_test('WindyGridworldKingsNoOp-v0', discount=1.0)

    def test_windy_gridworld_kings_stochastic(self):
        self._run_test('WindyGridworldKingsStochastic-v0', discount=1.0)

    def _run_test(self, env_id, discount):
        env = gym.make(env_id)
        V = value_iteration(env, discount)

        print(env_id + ':')
        if isinstance(env, Gridworld):
            # Print the values in the shape of the gridworld
            self._print_gridworld(env, V)
        else:
            # Just print a list of the encoded/decoded states and their values
            for s in env.states():
                print(s, ':', env._decode(s), ':', '{:.2f}'.format(V[s]))
        print(flush=True)

    def _print_gridworld(self, env, V):
        # First get the string length of the longest number
        formatter = lambda v: '{:+.2f}'.format(v)
        maxlen = 0
        for s in env._decoded_states():
            v = V[env._encode(s)]
            maxlen = max(maxlen, len(formatter(v)))

        # Now we can actually print the values
        for y in reversed(range(env._dims[1])):
            for x in range(env._dims[0]):
                state = (x, y)
                if not env._is_blocked(state):
                    s = env._encode(state)
                    print(formatter(V[s]).rjust(maxlen), end=' ' * 2)
                else:
                    print(' ' * maxlen, end=' ' * 2)
            print()
