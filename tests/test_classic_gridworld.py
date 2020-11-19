import unittest

import gym

import sutton_barto_gym


class TestClassicGridworld(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('ClassicGridworld-v0')
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
        for _ in range(5):
            _, reward, _, _ = env.step(0)
            self.assertEqual(reward, 0.0)
