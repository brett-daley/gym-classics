import numpy as np
from scipy.stats import poisson

from gym_classics.envs.abstract.base_env import BaseEnv


class JacksCarRental(BaseEnv):
    """Jack's Car Rental problem converted into an episodic task.

    States are 2-tuples of the number of cars at both locations.

    Page 81 of Sutton & Barto (2018, 2nd ed.).
    """

    def __init__(self):
        # Poission distributions for requests and dropoffs at both locations
        self._loc1_requests_distr = TruncatedPoisson(3)
        self._loc1_dropoffs_distr = TruncatedPoisson(3)
        self._loc2_requests_distr = TruncatedPoisson(4)
        self._loc2_dropoffs_distr = TruncatedPoisson(2)

        # Episode terminates after 100 days (timesteps)
        self._t = 0
        self._time_limit = 100

        # Bypass the search for reachable states because we know the whole grid is valid
        states = {(i, j) for i in range(21) for j in range(21)}
        super().__init__(starts={(10, 10)}, n_actions=11, reachable_states=states)

    def seed(self, seed=None):
        seeds = super().seed(seed)
        # Make sure each distribution has access to the np_random module
        for distr in [self._loc1_requests_distr, self._loc1_dropoffs_distr,
                      self._loc2_requests_distr, self._loc2_dropoffs_distr]:
            distr.np_random = self.np_random
        return seeds

    def reset(self):
        self._t = 0
        return super().reset()

    def step(self, action):
        self._t += 1
        return super().step(action)

    def _sample_random_elements(self, state, action):
        loc1_requests = self._loc1_requests_distr.sample()
        loc1_dropoffs = self._loc1_dropoffs_distr.sample()
        loc2_requests = self._loc2_requests_distr.sample()
        loc2_dropoffs = self._loc2_dropoffs_distr.sample()
        requests = [loc1_requests, loc2_requests]
        dropoffs = [loc1_dropoffs, loc2_dropoffs]
        return (requests, dropoffs)

    def _deterministic_step(self, state, action, requests, dropoffs):
        next_state, prob, moved_cars, fulfilled_requests = self._next_state(state, action, requests, dropoffs)
        reward = self._reward(moved_cars, fulfilled_requests)
        done = self._done()
        if done:
            next_state = state
        return next_state, reward, done, prob

    def _next_state(self, state, action, requests, dropoffs):
        # Compute the probability of this requests/dropoffs pair
        prob = self._loc1_requests_distr.pmf(requests[0]) \
                * self._loc1_dropoffs_distr.pmf(dropoffs[0]) \
                * self._loc2_requests_distr.pmf(requests[1]) \
                * self._loc2_dropoffs_distr.pmf(dropoffs[1])

        # Convert the action to a +/- delta representing the cars moved from loc 1 to 2
        action -= 5
        # We can't move more cars than are available at the source location
        if action >= 0:
            moved_cars = min(state[0], action)
        else:
            moved_cars = -min(state[1], abs(action))

        # Move cars
        state = list(state)
        state[0] = max(state[0] - moved_cars, 0)
        state[1] = min(state[1] + moved_cars, 25)

        for i in range(2):  # For each location
            # Fulfill requests
            requests[i] = min(state[i], requests[i])
            state[i] -= requests[i]

            # Fulfill dropoffs
            state[i] = min(state[i] + dropoffs[i], 20)

        fulfilled_requests = sum(requests)
        return tuple(state), prob, moved_cars, fulfilled_requests

    def _reward(self, moved_cars, fulfilled_requests):
        return -2.0 * abs(moved_cars) + 10.0 * fulfilled_requests

    def _done(self):
        return self._t == self._time_limit

    def _generate_transitions(self, state, action):
        for loc1_requests in self._loc1_requests_distr.domain():
            for loc1_dropoffs in self._loc1_dropoffs_distr.domain():
                for loc2_requests in self._loc2_requests_distr.domain():
                    for loc2_dropoffs in self._loc2_dropoffs_distr.domain():
                        requests = [loc1_requests, loc2_requests]
                        dropoffs = [loc1_dropoffs, loc2_dropoffs]
                        yield self._deterministic_step(state, action, requests, dropoffs)


class TruncatedPoisson:
    def __init__(self, mean, precision=0.1):
        assert isinstance(mean, int) and mean > 0
        assert 0.0 < precision < 1.0
        distr = poisson(mean)

        # Find the largest i such that 1 - sum_i Pr[i] < precision
        self.max = 0
        while 1.0 - distr.cdf(self.max) > precision:
            self.max += 1

        # Pre-compute the probability table and renormalize the sum to 1
        self.Pr = np.asarray([distr.pmf(i) for i in self.domain()])
        assert np.allclose(self.Pr.sum(), 1.0, rtol=0.0, atol=precision)
        self.Pr /= self.Pr.sum()

        # Save the domain as a list for efficient sampling
        self.values = list(self.domain())

    def domain(self):
        return range(self.max + 1)

    def pmf(self, i):
        return self.Pr[i]

    def sample(self):
        return self.np_random.choice(self.values, p=self.Pr)
