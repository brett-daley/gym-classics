from sutton_barto_gym.envs.abstract.gridworld import Gridworld


class DynaMaze(Gridworld):
    """Dyna Maze.

    Page 165 of Sutton & Barto (2018).
    """

    def __init__(self):
        blocks = frozenset({(2, 2), (2, 3), (2, 4), (5, 1), (7, 3), (7, 4), (7, 5)})
        super().__init__(dims=(9, 6), start=(0, 3), blocks=blocks)
        self._goal = (8, 5)

    def _reward(self, state, action, next_state):
        return 1.0 if next_state == self._goal else 0.0

    def _done(self, state, action, next_state):
        return next_state == self._goal
