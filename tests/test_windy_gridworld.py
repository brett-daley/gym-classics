import unittest

import gym

import sutton_barto_gym


class TestWindyGridworld(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('WindyGridworld-v0')
        self.env.seed(0)

    def test_random(self):
        env = self.env
        env.reset()
        for _ in range(1_000):
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            if done:
                env.reset()

    def test_reward(self):
        env = self.env
        env.reset()
        for _ in range(10):
            _, reward, _, _ = env.step(0)
            self.assertEqual(reward, -1.0)

    def test_optimal_path(self):
        env = self.env
        state = env.reset()

        for _ in range(9):
            state, reward, done, _ = env.step(1)
        for _ in range(4):
            env.step(2)
        env.step(3)
        state, reward, done, _ = env.step(3)

        self.assertEqual(env._decode(state), (7, 3))
        self.assertEqual(reward, 0.0)
        self.assertTrue(done)
