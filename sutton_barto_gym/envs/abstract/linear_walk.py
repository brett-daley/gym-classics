from sutton_barto_gym.envs.abstract.base_env import BaseEnv


class LinearWalk(BaseEnv):
    """Abstract class for creating 1-dimensional linear walks.

    Actions:
        0: left
        1: right
    """

    def __init__(self, length, left_reward, right_reward):
        """Instantiates a linear walk environment.

        Args:
            length
            start
            rewards
        """
        self._length = length
        self._left_reward = left_reward
        self._right_reward = right_reward

        assert length % 2 == 1
        self._start = length // 2

        super().__init__(n_actions=2)

        self._state = None  # Integer representing agent's position

    def reset(self):
        self._state = self._start
        return self._start

    def _decoded_states(self):
        return range(self._length)

    def _next_state(self, state, action):
        state += [-1, 1][action]
        return min(max(state, 0), self._length - 1)

    def _reward(self, state, action, next_state):
        sa_pair = (state, action)
        return {
            (0, 0):                self._left_reward,
            (self._length - 1, 1): self._right_reward,
        }.get(sa_pair, 0.0)

    def _done(self, state, action, next_state):
        sa_pair = (state, action)
        return sa_pair in {(0, 0), (self._length - 1, 1)}

    def _generate_transitions(self, state, action):
        next_state = self._next_state(state, action)
        reward = self._reward(state, action, next_state)
        prob = 1.0
        done = self._done(state, action, next_state)
        yield next_state, reward, prob, done
