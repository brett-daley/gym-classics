import unittest

import gym

import sutton_barto_gym


class TestEnvs(unittest.TestCase):
    def test_classic_gridworld(self):
        self._run_test('ClassicGridworld-v0')

    def test_windy_gridworld(self):
        self._run_test('WindyGridworld-v0')

    def test_windy_gridworld_optimal_path(self):
        env = gym.make('WindyGridworld-v0')
        env.seed(0)
        state = env.reset()

        for _ in range(9):
            env.step(1)
        for _ in range(4):
            env.step(2)
        for _ in range(2):
            state, reward, done, _ = env.step(3)

        self.assertEqual(env._decode(state), (7, 3))
        self.assertEqual(reward, 0.0)
        self.assertTrue(done)

    def _run_test(self, env_id):
        env = gym.make(env_id)
        env.seed(0)
        env.reset()

        for _ in range(1_000):
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            if done:
                env.reset()
