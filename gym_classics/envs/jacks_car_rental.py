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

        # Precompute the factored transition and reward functions for both locations
        self.P1, self.R1 = open_to_close(self._loc1_requests_distr, self._loc1_dropoffs_distr)
        self.P2, self.R2 = open_to_close(self._loc2_requests_distr, self._loc2_dropoffs_distr)

        # Episode terminates after 100 days (timesteps)
        self._t = 0
        self._time_limit = 100

        # Bypass the search for reachable states because we know the whole grid is valid
        states = [(i, j) for i in range(21) for j in range(21)]
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

    def _deterministic_step(self, state, action, next_state):
        # Convert the action to a +/- delta representing the cars moved from lot 1 to 2
        action -= 5

        # Move cars (we can't move more cars than are available at the source lot)
        state = list(state)
        moved_cars = clip(action, -state[1], state[0])
        state[0] -= moved_cars
        state[1] += moved_cars

        # Both lots evolve independently so we can multiply these to get the transition probability
        prob = self.P1[state[0]][next_state[0]] * self.P2[state[1]][next_state[1]]

        reward = self._reward(state, action)
        done = (self._t == self._time_limit)
        if done:
            next_state = state
        return tuple(next_state), reward, done, prob

    def _reward(self, state, action):
        # Reward = (10 * expected requests - 2 * attempted moves)
        # Note that this implicitly discourages the agent from trying to move more cars
        # than are available, which makes the optimal action unambiguous
        return -2.0 * abs(action) + self.R1[state[0]] + self.R2[state[1]]

    # We need to override these abstract methods but we don't actually use them
    def _next_state(self):
        pass
    def _done(self):
        pass

    def _generate_transitions(self, state, action):
        for next_state in self.states():
            next_state = self._decode(next_state)
            yield self._deterministic_step(state, action, next_state)


class JacksCarRentalModified(JacksCarRental):
    """Jack's Car Rental problem with two modifications to the reward function:
        1. One of Jack's employees can move a car from lot 1 to 2 for free
        2. Overnight parking costs $4 per lot with more than 10 cars

    Page 82, Exercise 4.7 of Sutton & Barto (2018, 2nd ed.).
    """
    def _reward(self, state, action):
        reward = super()._reward(state, action)

        # Jack's employee can move a car from lot 1 to lot 2 for free, so we save $2
        # whenever at least one car is moved to lot 2
        if action > 0:
            reward += 2.0

        # Jack has to pay for overnight parking: $4 per lot with more than 10 cars
        for i in range(2):
            if state[i] > 10:
                reward -= 4.0

        return reward


class TruncatedPoisson:
    def __init__(self, mean, threshold=1e-3):
        assert isinstance(mean, int) and mean > 0
        assert 0.0 < threshold < 1.0
        distr = poisson(mean)

        # Find the largest i such that Pr[i] > threshold
        self.max = 0
        while distr.pmf(self.max + 1) > threshold:
            self.max += 1

        # Save the domain as a list for efficient sampling
        self.domain = list(range(self.max + 1))

        # Pre-compute the probability table
        self.Pr = [distr.pmf(i) for i in self.domain]
        # Normalize so we can sample with np.random.choice
        # TODO: doesn't this mismatch create bias between learning vs DP?
        self.norm_Pr = np.asarray(self.Pr)
        self.norm_Pr /= self.norm_Pr.sum()

    def __iter__(self):
        return zip(self.domain, self.Pr)

    def sample(self):
        return self.np_random.choice(self.domain, p=self.norm_Pr)


def clip(x, low, high):
    """A scalar version of numpy.clip. Much faster because it avoids memory allocation."""
    return min(max(x, low), high)


def open_to_close(requests_distr, dropoffs_distr):
    """Calculates the transition function P and the reward function R over the two
    Poisson distributions: i.e. requests and dropoffs. Since the Poisson distribution's
    domain is infinite, the calculation is terminated within the given precision."""
    P = np.zeros((26, 21), dtype=np.float32)
    R = np.zeros(26)

    # How many cars were requested
    for requests, request_prob in requests_distr:
        # We can have up to 25 starting cars (20 capacity + 5 sent over)
        for n in range(26):
            # Expected reward: 10 * expected number rented out
            R[n] += (10.0 * request_prob * min(requests, n))

        # How many cars were returned
        for dropoffs, dropoff_prob in dropoffs_distr:
            for n in range(26):
                # We can satisfy as many requests as we have cars available
                satisfied_requests = min(requests, n)
                # Can't have more than 20 cars at the end of the day
                new_n = clip(n + dropoffs - satisfied_requests, 0, 20)
                # Increment the transition probability
                P[n][new_n] += request_prob * dropoff_prob

    return P, R
