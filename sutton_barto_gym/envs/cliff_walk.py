from sutton_barto_gym.envs.abstract.gridworld import Gridworld


class CliffWalk(Gridworld):
    """Cliff Walking.

    Page 132 of Sutton & Barto (2018).
    """

    def __init__(self):
        self._cliff = frozenset((x, 0) for x in range(1, 11))
        self._goal = (11, 0)
        super().__init__(dims=(12, 4), start=(0, 0))

    def _decoded_states(self):
        for s in super()._decoded_states():
            if (s != self._goal) and (s not in self._cliff):
                yield s

    def step(self, action):
        assert self.action_space.contains(action)
        state = self._state
        next_state = self._state = self._next_state(state, action)
        reward = self._reward(state, action, next_state)
        done = self._done(state, action, next_state)

        # TODO: find a way to remove this conditional and inherit from the super class
        if done:
            next_state = state

        return self._encode(next_state), reward, done, {}

    def _reward(self, state, action, next_state):
        return -100.0 if next_state in self._cliff else -1.0

    def _done(self, state, action, next_state):
        return (next_state == self._goal) or (next_state in self._cliff)

    def _generate_transitions(self, state, action):
        state = self._decode(state)
        next_state = self._next_state(state, action)
        reward = self._reward(state, action, next_state)
        prob = 1.0
        done = self._done(state, action, next_state)

        # TODO: find a way to remove this conditional and inherit from the super class
        if done:
            next_state = state

        yield self._encode(next_state), reward, prob, done
