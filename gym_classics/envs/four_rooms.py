from gym_classics.envs.abstract.noisy_gridworld import NoisyGridworld


class FourRooms(NoisyGridworld):
    def __init__(self):
        blocks = frozenset({(5,0), (5,2), (5,3), (5,4), (5,5), (5,6), (5,7), (5,9), (5,10),
                            (0,5), (2,5), (3,5), (4,5),
                            (6,4), (7,4), (9,4), (10,4)})
        self._goal = (10, 10)
        super().__init__(dims=(11, 11), start=(0, 0), blocks=blocks)

    def _reward(self, state, action, next_state):
        return 1.0 if self._done(state, action, next_state) else 0.0

    def _done(self, state, action, next_state):
        return next_state == self._goal
