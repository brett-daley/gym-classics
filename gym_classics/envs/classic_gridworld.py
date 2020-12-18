from gym_classics.envs.abstract.noisy_gridworld import NoisyGridworld


class ClassicGridworld(NoisyGridworld):
    """'Classic' Gridworld.

    Norvig & Russel (2020, 4th ed.).
    """

    def __init__(self):
        super().__init__(dims=(4, 3), starts={(0, 0)}, blocks=frozenset({(1, 1)}))

    def _reward(self, state, action, next_state):
        return {(3, 1): -1.0, (3, 2): 1.0}.get(state, 0.0)

    def _done(self, state, action, next_state):
        return state in {(3, 1), (3, 2)}
