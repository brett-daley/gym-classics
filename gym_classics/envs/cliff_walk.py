from gym_classics.envs.abstract.gridworld import Gridworld


class CliffWalk(Gridworld):
    """The Cliff Walking task used to contrast Sarsa and Q-Learning. The agent begins
    in the bottom-left cell and must navigate to the goal (bottom-right cell) without
    entering the region labeled "The Cliff."
    Reference: cite{3} (page 132, example 6.6).

    **states**: Grid cells.

    **actions**: Move up/right/down/left.

    **rewards**: -100 for entering The Cliff. -1 for all other transitions.

    **termination**: Entering The Cliff or reaching the goal.
    """

    def __init__(self):
        self._cliff = frozenset((x, 0) for x in range(1, 11))
        self._goal = (11, 0)
        super().__init__(dims=(12, 4), starts={(0, 0)})

    def _reward(self, state, action, next_state):
        return -100.0 if next_state in self._cliff else -1.0

    def _done(self, state, action, next_state):
        return (next_state == self._goal) or (next_state in self._cliff)
