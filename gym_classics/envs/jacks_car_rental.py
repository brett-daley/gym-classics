import numpy as np
from scipy.stats import poisson

from gym_classics.envs.abstract.base_env import BaseEnv
from gym_classics.utils import clip


class JacksCarRental(BaseEnv):
    """Jack's Car Rental problem converted into an episodic task.

    States are 2-tuples of the number of cars at both parking lots.

    Page 81 of Sutton & Barto (2018, 2nd ed.).
    """

    def __init__(self):
        # Poission distributions for requests and dropoffs at both lots
        self._lot1_requests_distr = TruncatedPoisson(3)
        self._lot1_dropoffs_distr = TruncatedPoisson(3)
        self._lot2_requests_distr = TruncatedPoisson(4)
        self._lot2_dropoffs_distr = TruncatedPoisson(2)

        # Precompute the factored transition and reward functions for both lots
        self.P1, self.R1 = open_to_close(self._lot1_requests_distr, self._lot1_dropoffs_distr)
        self.P2, self.R2 = open_to_close(self._lot2_requests_distr, self._lot2_dropoffs_distr)

        # Episode terminates after 100 days (timesteps)
        self._t = 0
        self._time_limit = 100

        # Bypass the search for reachable states because we know the whole grid is valid
        states = [(i, j) for i in range(21) for j in range(21)]
        super().__init__(starts={(10, 10)}, n_actions=11, reachable_states=states)

    def seed(self, seed=None):
        seeds = super().seed(seed)
        # Make sure each distribution has access to the np_random module
        for distr in [self._lot1_requests_distr, self._lot1_dropoffs_distr,
                      self._lot2_requests_distr, self._lot2_dropoffs_distr]:
            distr.np_random = self.np_random
        return seeds

    def reset(self):
        self._t = 0
        return super().reset()

    def step(self, action):
        self._t += 1
        assert self.action_space.contains(action)
        state = self._state
        action = decode_action(action)

        next_state = move_cars(state, action)

        requests, dropoffs = self._sample_random_elements()
        for i in range(len(next_state)):
            next_state[i] = handle_requests_and_dropoffs(next_state[i], requests[i], dropoffs[i])

        next_state, reward, done, _ = self._deterministic_step(state, action, next_state)
        self._state = next_state
        return self._encode(next_state), reward, done, {}

    def _sample_random_elements(self):
        lot1_requests = self._lot1_requests_distr.sample()
        lot1_dropoffs = self._lot1_dropoffs_distr.sample()
        lot2_requests = self._lot2_requests_distr.sample()
        lot2_dropoffs = self._lot2_dropoffs_distr.sample()
        requests = [lot1_requests, lot2_requests]
        dropoffs = [lot1_dropoffs, lot2_dropoffs]
        return (requests, dropoffs)

    def _deterministic_step(self, state, action, next_state):
        state_after_move = move_cars(state, action)

        # Both lots evolve independently so we can multiply these to get the transition probability
        prob = self.P1[state_after_move[0]][next_state[0]] \
             * self.P2[state_after_move[1]][next_state[1]]

        reward = self._reward(state_after_move, action)
        done = self._done()
        if done:
            next_state = state
        return tuple(next_state), reward, done, prob

    def _next_state(self):
        # We need to override this abstract method but we don't actually use it
        raise NotImplementedError

    def _reward(self, state_after_move, action):
        # Reward = (10 * expected requests - 2 * attempted moves)
        # Note that this implicitly discourages the agent from trying to move more cars
        # than are available, which makes the optimal action unambiguous
        n1, n2 = state_after_move
        return -2.0 * abs(action) + self.R1[n1] + self.R2[n2]

    def _done(self):
        return self._t == self._time_limit

    def _generate_transitions(self, state, action):
        action = decode_action(action)
        for next_state in self.states():
            next_state = self._decode(next_state)
            yield self._deterministic_step(state, action, next_state)


class JacksCarRentalModified(JacksCarRental):
    """Jack's Car Rental problem with two modifications to the reward function:
        1. One of Jack's employees can move a car from lot 1 to 2 for free
        2. Overnight parking costs $4 per lot with more than 10 cars

    Page 82, Exercise 4.7 of Sutton & Barto (2018, 2nd ed.).
    """
    def _reward(self, state_after_move, action):
        reward = super()._reward(state_after_move, action)

        # Jack's employee can move a car from lot 1 to lot 2 for free, so we save $2
        # whenever at least one car is moved to lot 2
        if action > 0:
            reward += 2.0

        # Jack has to pay for overnight parking: $4 per lot with more than 10 cars
        for i in range(2):
            if state_after_move[i] > 10:
                reward -= 4.0

        return reward


class TruncatedPoisson:
    def __init__(self, mean, threshold=1e-6):
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
        self.Pr = np.asarray([distr.pmf(i) for i in self.domain])
        # Normalize so we can sample with np.random.choice
        self.Pr /= self.Pr.sum()

    def __iter__(self):
        return zip(self.domain, self.Pr)

    def sample(self):
        return self.np_random.choice(self.domain, p=self.Pr)


def decode_action(i):
    # Convert the integer to a +/- delta representing the cars moved from lot 1 to 2
    return i - 5


def move_cars(state, action):
    # We can't move more cars than are available at the source lot
    moved_cars = clip(action, -state[1], state[0])
    return [state[0] - moved_cars, state[1] + moved_cars]


def handle_requests_and_dropoffs(cars, requests, dropoffs):
    # We can satisfy as many requests as we have cars available
    satisfied_requests = min(cars, requests)
    # Can't have more than 20 cars at the end of the day
    return clip(cars + dropoffs - satisfied_requests, 0, 20)


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
                new_n = handle_requests_and_dropoffs(n, requests, dropoffs)
                # Increment the transition probability
                P[n][new_n] += request_prob * dropoff_prob

    return P, R
