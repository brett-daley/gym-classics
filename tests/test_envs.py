import unittest

import gym

import gym_classics


class TestEnvs(unittest.TestCase):
    def test_5walk(self):
        self._test_interface('5Walk-v0')

    def test_5walk_optimal_path(self):
        env = gym.make('5Walk-v0')
        env.reset()

        for _ in range(3):
            state, reward, done, _ = env.step(1)

        self.assertEqual(state, 4)
        self.assertEqual(reward, 1.0)
        self.assertTrue(done)


    def test_19walk(self):
        self._test_interface('19Walk-v0')

    def test_19walk_optimal_path(self):
        env = gym.make('19Walk-v0')
        env.reset()

        for _ in range(10):
            state, reward, done, _ = env.step(1)

        self.assertEqual(state, 18)
        self.assertEqual(reward, 1.0)
        self.assertTrue(done)


    def test_classic_gridworld(self):
        self._test_interface('ClassicGridworld-v0')


    def test_cliff_walk(self):
        self._test_interface('CliffWalk-v0')

    def test_cliff_walk_terminal(self):
        env = gym.make('CliffWalk-v0')
        env.reset()

        state, reward, done, _ = env.step(1)

        self.assertEqual(env._decode(state), (0, 0))
        self.assertEqual(reward, -100.0)
        self.assertTrue(done)

    def test_cliff_walk_optimal_path(self):
        env = gym.make('CliffWalk-v0')
        env.reset()

        env.step(0)
        for _ in range(11):
            env.step(1)
        state, reward, done, _ = env.step(2)

        self.assertEqual(env._decode(state), (11, 1))
        self.assertEqual(reward, -1.0)
        self.assertTrue(done)


    def test_dyna_maze(self):
        self._test_interface('DynaMaze-v0')

    def test_dyna_maze_optimal_path(self):
        env = gym.make('DynaMaze-v0')
        env.reset()

        env.step(1)
        for _ in range(2):
            env.step(2)
        for _ in range(2):
            env.step(1)
        env.step(0)
        for _ in range(5):
            env.step(1)
        for _ in range(3):
            state, reward, done, _ = env.step(0)

        self.assertEqual(env._decode(state), (8, 4))
        self.assertEqual(reward, 1.0)
        self.assertTrue(done)


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

    def test_windy_gridworld_optimal_path(self):
        env = gym.make('WindyGridworld-v0')
        env.reset()

        for _ in range(9):
            env.step(1)
        for _ in range(4):
            env.step(2)
        for _ in range(2):
            state, reward, done, _ = env.step(3)

        self.assertEqual(env._decode(state), (8, 2))
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

        self.assertEqual(env._decode(state), (6, 1))
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

        self.assertEqual(env._decode(state), (6, 1))
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
