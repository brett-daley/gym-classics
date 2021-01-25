from gym_classics.envs.abstract.noisy_gridworld import NoisyGridworld


class SparseGridworld(NoisyGridworld):
    """A 10x8 featureless gridworld. The agent starts in the mid-left cell, and the goal
    is slightly to the right of center. To make it more challenging, the same 80-10-10
    transition probabilities from `ClassicGridworld` are used. Great for testing various
    forms of credit assignment in the presence of noise.
    Reference: cite{3} (page 147, figure 7.4)

    **states:** Agent cell location.

    **actions:** Move up/right/down/left.

    **rewards:** +1 for reaching the goal.

    **termination:** Reaching the goal.
    """

    def __init__(self):
        self._goal = (6, 3)
        super().__init__(dims=(10, 8), starts={(1, 3)})

    def _reward(self, state, action, next_state):
        return 1.0 if self._done(state, action, next_state) else 0.0

    def _done(self, state, action, next_state):
        return next_state == self._goal
