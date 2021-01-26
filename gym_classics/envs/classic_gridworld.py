from gym_classics.envs.abstract.noisy_gridworld import NoisyGridworld


class ClassicGridworld(NoisyGridworld):
    """A 4x3 pedagogical gridworld. The agent starts in the bottom-left cell. Actions
    are noisy; with a 10% chance each, a move action may be rotated by 90 degrees
    clockwise or counter-clockwise (the "80-10-10 rule"). Cell (1, 1) is blocked and
    cannot be occupied by the agent.

    **reference:** cite{1} (page 646).

    **state**: Grid location.

    **actions**: Move up/right/down/left.

    **rewards**: +1 for taking any action in cell (3, 2). -1 for taking any
    action in cell (3, 1). *NOTE:* The original version adds a -0.04 penalty to all other
    transitions, but this implementation does not.

    **termination**: Earning a nonzero reward.
    """

    def __init__(self):
        super().__init__(dims=(4, 3), starts={(0, 0)}, blocks=frozenset({(1, 1)}))

    def _reward(self, state, action, next_state):
        return {(3, 1): -1.0, (3, 2): 1.0}.get(state, 0.0)

    def _done(self, state, action, next_state):
        return state in {(3, 1), (3, 2)}
