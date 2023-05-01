import unittest

import gymnasium as gym

import gym_classics
gym_classics.register('gymnasium')


class TestGymnasium(unittest.TestCase):
    def test_5walk(self):
        self._test_interface('5Walk-v0')

    def test_19walk(self):
        self._test_interface('19Walk-v0')

    def test_classic_gridworld(self):
        self._test_interface('ClassicGridworld-v0')

    def test_cliff_walk(self):
        self._test_interface('CliffWalk-v0')

    def test_dyna_maze(self):
        self._test_interface('DynaMaze-v0')

    def test_four_rooms(self):
        self._test_interface('FourRooms-v0')

    def test_jacks_car_rental(self):
        self._test_interface('JacksCarRental-v0')

    def test_jacks_car_rental_modified(self):
        self._test_interface('JacksCarRentalModified-v0')

    # def test_racetrack1(self):
    #     self._test_interface('Racetrack1-v0')

    # def test_racetrack2(self):
    #     self._test_interface('Racetrack2-v0')

    def test_sparse_gridworld(self):
        self._test_interface('SparseGridworld-v0')

    def test_windy_gridworld(self):
        self._test_interface('WindyGridworld-v0')

    def test_windy_gridworld_kings(self):
        self._test_interface('WindyGridworldKings-v0')

    def test_windy_gridworld_kings_no_op(self):
        self._test_interface('WindyGridworldKingsNoOp-v0')

    def test_windy_gridworld_stochastic(self):
        self._test_interface('WindyGridworldKingsStochastic-v0')

    def _test_interface(self, env_id):
        env = gym.make(env_id)
        _, _ = env.reset(seed=0)

        for _ in range(1_000):
            action = env.action_space.sample()
            _, _, done, _, _ = env.step(action)
            if done:
                env.reset()
