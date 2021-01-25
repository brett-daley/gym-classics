import unittest

import gym
import numpy as np

import gym_classics


class TestModel(unittest.TestCase):
    def test_5walk(self):
        self._run_test('5Walk-v0', discount=0.9, deterministic=True)

    def test_19walk(self):
        self._run_test('19Walk-v0', discount=0.9, deterministic=True)

    def test_classic_gridworld(self):
        self._run_test('ClassicGridworld-v0', discount=0.9)

    def test_cliff_walk(self):
        self._run_test('CliffWalk-v0', discount=0.9, deterministic=True)

    def test_dyna_maze(self):
        self._run_test('DynaMaze-v0', discount=0.95, deterministic=True)

    def test_four_rooms(self):
        self._run_test('FourRooms-v0', discount=0.95)

    def test_sparse_gridworld(self):
        self._run_test('SparseGridworld-v0', discount=0.9)

    def test_windy_gridworld(self):
        self._run_test('WindyGridworld-v0', discount=1.0, deterministic=True)

    def test_windy_gridworld_kings(self):
        self._run_test('WindyGridworldKings-v0', discount=1.0, deterministic=True)

    def test_windy_gridworld_kings_no_op(self):
        self._run_test('WindyGridworldKingsNoOp-v0', discount=1.0, deterministic=True)

    def test_windy_gridworld_kings_stochastic(self):
        self._run_test('WindyGridworldKingsStochastic-v0', discount=1.0)


    # NOTE: the below tests are intentionally commented out because they're slow to run

    # def test_jacks_car_rental(self):
    #     self._run_test('JacksCarRental-v0', discount=0.9)

    # def test_jacks_car_rental_modified(self):
    #     self._run_test('JacksCarRentalModified-v0', discount=0.9)

    # TODO: possible bug in reward/done functions
    # def test_racetrack1(self):
    #     self._run_test('Racetrack1-v0', discount=0.9)

    # TODO: possible bug in reward/done functions
    # def test_racetrack2(self):
    #     self._run_test('Racetrack2-v0', discount=0.9)

    ###################################################################################


    def _run_test(self, env_id, discount, deterministic=False):
        np.random.seed(0)

        env = gym.make(env_id)
        env._use_sparse_model = True
        env.seed(0)

        for s in env.states():
            for a in env.actions():
                # We don't need to sample more than once if the env is deterministic
                n = 1000 if not deterministic else 1

                states1, rewards1, dones1, probs1 = self._approximate_model(env, s, a, n)
                states2, rewards2, dones2, probs2 = env.model(s, a)

                # States and dones should be the same
                self.assertTrue((states1 == states2).all())

                # Rewards/dones should also be the same, but there may be extremely rare
                # transitions that will not be sampled (e.g. Jack's Car Rental).
                # Instead, just make sure that all of the sampled transitions are correct.
                nonnan = np.logical_not(np.isnan(rewards1))
                self.assertTrue((rewards1[nonnan] == rewards2[nonnan]).all())
                self.assertTrue((dones1[nonnan] == dones2[nonnan]).all())

                if not deterministic:
                    # Make sure that any impossible transition according to the model
                    # is never sampled by the step() method
                    zero = (probs2 == 0.0)
                    self.assertTrue((probs1[zero] == 0.0).all())

                    # Make sure nonzero probabilities are close (allowing for variance)
                    nonzero = np.logical_not(zero)
                    try:
                        self.assertTrue(np.allclose(probs1[nonzero], probs2[nonzero], atol=0.05))
                    except AssertionError:
                        print(probs1)
                        print(probs2)
                        print(probs1 - probs2)
                        raise
                else:
                    # The environment is deterministic, so these should be identical
                    self.assertTrue((probs1 == probs2).all())

    def _approximate_model(self, env, state, action, n):
        S = env.observation_space.n
        next_states = np.arange(S)
        dones = np.nan * np.ones(S)
        rewards = np.nan * np.ones(S)
        counts = np.zeros(S)

        for _ in range(n):
            # Reset environment state each time
            env._state = env._decode(state)

            # Sample an outcome from this state-action pair
            ns, r, d, _ = env.step(action)

            dones[ns] = float(d)
            rewards[ns] = r
            counts[ns] += 1.0

            if hasattr(env, '_t'):
                # Reset the timer if the environment has one
                env._t = 0

        # Convert the counts to probabilities
        probabilities = counts / counts.sum()
        np.nan_to_num(probabilities, copy=False)

        return (next_states, rewards, dones, probabilities)
