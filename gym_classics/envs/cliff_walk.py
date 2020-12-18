from gym_classics.envs.abstract.gridworld import Gridworld


class CliffWalk(Gridworld):
    """Cliff Walking.

    Page 132 of Sutton & Barto (2018).
    """

    def __init__(self):
        self._cliff = frozenset((x, 0) for x in range(1, 11))
        self._goal = (11, 0)
        super().__init__(dims=(12, 4), start=(0, 0))

    def _reward(self, state, action, next_state):
        return -100.0 if next_state in self._cliff else -1.0

    def _done(self, state, action, next_state):
        return (next_state == self._goal) or (next_state in self._cliff)
