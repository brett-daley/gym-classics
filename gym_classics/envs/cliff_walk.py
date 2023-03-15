from gym_classics.envs.abstract.gridworld import Gridworld


class CliffWalk(Gridworld):
    """The Cliff Walking task, a 12x4 gridworld often used to contrast Sarsa with
    Q-Learning. The agent begins in the bottom-left cell and must navigate to the goal
    (bottom-right cell) without entering the region along the bottom ("The Cliff").

    **reference:** cite{3} (page 132, example 6.6).

    **state**: Grid location.

    **actions**: Move up/right/down/left.

    **rewards**: -100 for entering The Cliff. -1 for all other transitions.

    **termination**: Entering The Cliff or reaching the goal.
    """

    layout = """
|            |
|            |
|            |
|S          G|
"""

    def __init__(self):
        self._cliff = frozenset((x, 0) for x in range(1, 11))
        super().__init__(CliffWalk.layout)

    def _reward(self, state, action, next_state):
        return -100.0 if next_state in self._cliff else -1.0

    def _done(self, state, action, next_state):
        return (next_state in self._goals) or (next_state in self._cliff)
