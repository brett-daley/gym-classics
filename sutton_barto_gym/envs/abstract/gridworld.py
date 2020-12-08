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

    def __init__(self, dims, start, blocks=set(), goals=set()):
        """Instantiates a gridworld environment.

        Args:
            dims
            start
            blocks
            goals
        """
        self._dims = dims
        self._start = start
        self._blocks = blocks
        self._goals = goals

        super().__init__(n_actions=4)

        self._state = None  # Tuple representing agent's position

    def reset(self):
        self._state = self._start

    def actions(self):
        return range(self.action_space.n)

    def _decoded_states(self):
        for x in range(self._dims[0]):
            for y in range(self._dims[1]):
                state = (x, y)
                if not self._is_block(state):
                    yield state

    def step(self, action):
        assert self.action_space.contains(action)
        state = self._state
        next_state = self._state = self._apply_move(self._state, action)

        reward = self._reward(state, action, next_state)
        done = self._done(state, action, next_state)
        return self._encode(next_state), reward, done, {}

    def _apply_move(self, state, action):
        x, y = state
        new_state = {
            0: (x,   y+1),  # Up
            1: (x+1, y),    # Right
            2: (x,   y-1),  # Down
            3: (x-1, y)     # Left
        }[action]

        if self._is_block(new_state):
            return state

        return self._clamp(new_state)

    def _clamp(self, state):
        """Clamps the state within the grid dimensions."""
        x, y = state
        x = max(0, min(x, self._dims[0] - 1))
        y = max(0, min(y, self._dims[1] - 1))
        return (x, y)

    def _is_block(self, state):
        """Checks whether this state intersects a wall."""
        return state in self._blocks

    def _is_goal(self, state):
        """Checks whether this state is a goal."""
        return state in self._goals

    def _generate_transitions(self, state, action):
        state = self._decode(state)
        next_state = self._apply_move(state, action)
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
