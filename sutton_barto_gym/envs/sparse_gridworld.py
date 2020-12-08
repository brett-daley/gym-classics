from sutton_barto_gym.envs.abstract.noisy_gridworld import NoisyGridworld


class SparseGridworld(NoisyGridworld):
    """Dyna Maze.

    Page 147 of Sutton & Barto (2018).
    """

    def __init__(self):
        super().__init__(dims=(10, 8), start=(1, 3))
        self._goal = (6, 3)

    def _reward(self, state, action, next_state):
        if state == self._goal or next_state == self._goal:
            return 1.0
        return 0.0

    def _done(self, state, action, next_state):
        if state == self._goal or next_state == self._goal:
            return True
        return False
