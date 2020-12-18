from gym_classics.envs.abstract.base_env import BaseEnv


class LinearWalk(BaseEnv):
    """Abstract class for creating 1-dimensional linear walks.

    Actions:
        0: left
        1: right
    """

    def __init__(self, length, left_reward, right_reward):
        """Instantiates a linear walk environment.

        State is an integer representing agent's position.

        Args:
            length
            start
            rewards
        """
        self._length = length
        self._left_reward = left_reward
        self._right_reward = right_reward

        assert length % 2 == 1
        super().__init__(start=(length // 2), n_actions=2)

    def _next_state(self, state, action):
        state += [-1, 1][action]
        next_state = min(max(state, 0), self._length - 1)
        return next_state, 1.0

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
        yield self._deterministic_step(state, action)
