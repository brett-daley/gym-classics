from gym_classics.envs.abstract.noisy_gridworld import NoisyGridworld


class ClassicGridworld(NoisyGridworld):
    """A pedagogical 4x3 gridworld with the agent starting in the bottom-left cell.
    Actions are noisy; with a 10% chance each, a move action may rotated by 90
    degrees clockwise or counter-clockwise.
    Reference: cite{1} (page 646).

    **states**: Grid cells.

    **actions**: Move up/right/down/left.

    **rewards**: +1 for taking any action in the top-right cell. -1 for taking any
    action in the mid-right cell.

    **termination**: Earning a nonzero reward.
    """

    def __init__(self):
        super().__init__(dims=(4, 3), starts={(0, 0)}, blocks=frozenset({(1, 1)}))

    def _reward(self, state, action, next_state):
        return {(3, 1): -1.0, (3, 2): 1.0}.get(state, 0.0)

    def _done(self, state, action, next_state):
        return state in {(3, 1), (3, 2)}
