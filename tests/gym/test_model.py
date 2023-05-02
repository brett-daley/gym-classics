import unittest

import gym
import numpy as np

import gym_classics
gym_classics.register('gym')


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

    # def test_racetrack1(self):
    #     self._run_test('Racetrack1-v0', discount=0.9)

    # def test_racetrack2(self):
    #     self._run_test('Racetrack2-v0', discount=0.9)

    ###################################################################################


    def _run_test(self, env_id, discount, deterministic=False):
        env = gym.make(env_id)
        env.reset(seed=0)

        def make_sparse(indices, values):
            S = env.observation_space.n
            sparse_array = np.zeros(S, dtype=values.dtype)
            sparse_array[indices] = values
            return sparse_array

        for s in env.states():
            for a in env.actions():
                # We don't need to sample more than once if the env is deterministic
                n = 1250 if not deterministic else 1
                states1, rewards1, dones1, probs1, were_sampled = self._approximate_model(env, s, a, n)

                # Now get the real model outputs and pad them with zeros if needed (so all vectors have the same length)
                states2, rewards2, dones2, probs2 = env.model(s, a)
                indices = states2.copy()
                states2 = make_sparse(indices, states2)
                rewards2 = make_sparse(indices, rewards2)
                dones2 = make_sparse(indices, dones2)
                probs2 = make_sparse(indices, probs2)

                if deterministic:
                    # For deterministic environments, the models should match exactly.
                    self.assertTrue((states1 == states2).all())
                    self.assertTrue((rewards1 == rewards2).all())
                    self.assertTrue((dones1 == dones2).all())
                    self.assertTrue((probs1 == probs2).all())
                    continue

                # We might not sample all possible states in a stochastic environment, but the
                # samples we do obtain should at least be a subset of them.
                self.assertTrue(set(states1).issubset(set(states2)))

                # Rewards/dones should be the same, but there may be extremely rare
                # transitions that will not be sampled (e.g. Jack's Car Rental).
                # Instead, just make sure that all of the sampled transitions are correct.
                self.assertTrue((rewards1[were_sampled] == rewards2[were_sampled]).all())
                self.assertTrue((dones1[were_sampled] == dones2[were_sampled]).all())

                # Make sure that any impossible transition according to the model
                # is never sampled by the step() method.
                zero = (probs2 == 0.0)
                self.assertTrue((probs1[zero] == 0.0).all())

                # Make sure nonzero probabilities are somewhat close (allowing for variance).
                try:
                    self.assertTrue(np.allclose(probs1[were_sampled], probs2[were_sampled], atol=0.05))
                except AssertionError:
                    np.set_printoptions(threshold=np.inf)
                    print(probs1)
                    print(probs2)
                    print(probs1 - probs2)
                    raise

    def _approximate_model(self, env, state, action, n):
        S = env.observation_space.n
        next_states = np.zeros(S, dtype=np.int64)
        dones = np.zeros(S)
        rewards = np.zeros(S)
        counts = np.zeros(S)

        for _ in range(n):
            # Reset environment state each time
            env.unwrapped.state = env.decode(state)

            # Sample an outcome from this state-action pair
            ns, r, d, _, _ = env.step(action)

            next_states[ns] = ns
            dones[ns] = float(d)
            rewards[ns] = r
            counts[ns] += 1

        # Convert the counts to probabilities
        probabilities = counts / counts.sum()
        np.nan_to_num(probabilities, copy=False)

        were_sampled = (counts > 0)
        return (next_states, rewards, dones, probabilities, were_sampled)
