from gym_classics.envs.abstract.gridworld import Gridworld


class DynaMaze(Gridworld):
    """A 9x6 deterministic gridworld with barriers to make navigation more challenging.
    The agent starts in cell (0, 3); the goal is the top-right cell.

    **reference:** cite{3} (page 164, example 8.1).

    **state**: Grid location.

    **actions**: Move up/right/down/left.

    **rewards**: +1 for episode termination.

    **termination**: Reaching the goal.
    """

    layout = """
|       XG|
|  X    X |
|S X    X |
|  X      |
|     X   |
|         |
"""

    def __init__(self):
        super().__init__(DynaMaze.layout)

    def _reward(self, state, action, next_state):
        return 1.0 if next_state in self._goals else 0.0

    def _done(self, state, action, next_state):
        return next_state in self._goals
