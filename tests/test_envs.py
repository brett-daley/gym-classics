import unittest

import gym

import sutton_barto_gym


class TestEnvs(unittest.TestCase):
    def test_classic_gridworld(self):
        self._test_interface('ClassicGridworld-v0')


    def test_windy_gridworld(self):
        self._test_interface('WindyGridworld-v0')

    def test_windy_gridworld_optimal_path(self):
        env = gym.make('WindyGridworld-v0')
        env.reset()

        for _ in range(9):
            env.step(1)
        for _ in range(4):
            env.step(2)
        for _ in range(2):
            state, reward, done, _ = env.step(3)

        self.assertEqual(env._decode(state), (7, 3))
        self.assertEqual(reward, 0.0)
        self.assertTrue(done)


    def test_windy_gridworld_kings(self):
        self._test_interface('WindyGridworldKings-v0')

    def test_windy_gridworld_kings_optimal_path(self):
        env = gym.make('WindyGridworldKings-v0')
        env.reset()

        for _ in range(6):
            env.step(5)
        state, reward, done, _ = env.step(1)

        self.assertEqual(env._decode(state), (7, 3))
        self.assertEqual(reward, 0.0)
        self.assertTrue(done)


    def test_windy_gridworld_kings_no_op(self):
        self._test_interface('WindyGridworldKingsNoOp-v0')

    def test_windy_gridworld_kings_no_op_action(self):
        env = gym.make('WindyGridworldKingsNoOp-v0')
        init_state = env.reset()
        for _ in range(10):
            state, _, _, _ = env.step(8)
        self.assertEqual(state, init_state)

    def test_windy_gridworld_kings_no_op_optimal_path(self):
        env = gym.make('WindyGridworldKingsNoOp-v0')
        env.reset()

        for _ in range(6):
            env.step(5)
        state, reward, done, _ = env.step(1)

        self.assertEqual(env._decode(state), (7, 3))
        self.assertEqual(reward, 0.0)
        self.assertTrue(done)


    def test_windy_gridworld_stochastic(self):
        self._test_interface('WindyGridworldKingsStochastic-v0')


    def _test_interface(self, env_id):
        env = gym.make(env_id)
        env.seed(0)
        env.reset()

        for _ in range(1_000):
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            if done:
                env.reset()
