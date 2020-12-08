from abc import abstractmethod

from sutton_barto_gym.envs.abstract.base_env import BaseEnv


class Gridworld(BaseEnv):
    """Abstract class for creating gridworld-type environments.

    Actions:
        0: up
        1: right
        2: down
        3: left
    """

    def __init__(self, dims, start, blocks=set()):
        """Instantiates a gridworld environment.

        Args:
            dims
            start
            blocks
        """
        self._dims = dims
        self._start = start
        self._blocks = blocks

        super().__init__(n_actions=4)

        self._state = None  # Tuple representing agent's position

    def reset(self):
        self._state = self._start
        return self._encode(self._state)

    def actions(self):
        return range(self.action_space.n)

    def _decoded_states(self):
        for x in range(self._dims[0]):
            for y in range(self._dims[1]):
                state = (x, y)
                if not self._is_blocked(state):
                    yield state

    def step(self, action):
        assert self.action_space.contains(action)
        state = self._state
        next_state = self._state = self._take_action(state, action)
        reward = self._reward(state, action, next_state)
        done = self._done(state, action, next_state)
        return self._encode(next_state), reward, done, {}

    def _take_action(self, state, action):
        next_state = self._move(state, action)
        if self._is_blocked(next_state):
            next_state = state
        return self._clamp(next_state)

    def _move(self, state, action):
        x, y = state
        return {
            0: (x,   y+1),  # Up
            1: (x+1, y),    # Right
            2: (x,   y-1),  # Down
            3: (x-1, y)     # Left
        }[action]

    def _clamp(self, state):
        """Clamps the state within the grid dimensions."""
        x, y = state
        x = max(0, min(x, self._dims[0] - 1))
        y = max(0, min(y, self._dims[1] - 1))
        return (x, y)

    def _is_blocked(self, state):
        """Returns True if this state cannot be occupied, False otherwise."""
        return state in self._blocks

    def _generate_transitions(self, state, action):
        state = self._decode(state)
        next_state = self._take_action(state, action)
        reward = self._reward(state, action, next_state)
        prob = 1.0
        done = self._done(state, action, next_state)
        yield self._encode(next_state), reward, prob, done

    @abstractmethod
    def _reward(self, state, action, next_state):
        """Returns the reward yielded by this (S,A,S') outcome."""
        raise NotImplementedError

    @abstractmethod
    def _done(self, state, action, next_state):
        """Returns True if this (S,A,S') outcome should terminate, False otherwise."""
        raise NotImplementedError
