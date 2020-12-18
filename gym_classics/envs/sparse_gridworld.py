from gym_classics.envs.abstract.noisy_gridworld import NoisyGridworld


class SparseGridworld(NoisyGridworld):
    """Dyna Maze.

    Page 147 of Sutton & Barto (2018).
    """

    def __init__(self):
        self._goal = (6, 3)
        super().__init__(dims=(10, 8), start=(1, 3))

    def _reward(self, state, action, next_state):
        return 1.0 if self._done(state, action, next_state) else 0.0

    def _done(self, state, action, next_state):
        return next_state == self._goal
