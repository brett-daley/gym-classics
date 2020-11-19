from abc import abstractmethod

from gym.spaces import Discrete
import numpy as np

from sutton_barto_gym.building_blocks import BaseEnv


class AbstractGridworld(BaseEnv):
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

        self.observation_space = Discrete(np.prod(dims) - len(blocks))
        self.action_space = Discrete(4)

        self._state = None  # Tuple representing agent's position

        # Make look-up tables for quick state-to-integer conversion and vice-versa
        self._encoder = {}
        self._decoder = {}
        i = 0
        for x in range(dims[0]):
            for y in range(dims[1]):
                state = (x, y)
                if not self._is_block(state):
                    self._encoder[state] = i
                    self._decoder[i] = state
                    i += 1

    def reset(self):
        self._state = self._start

    def actions(self):
        return range(self.action_space.n)

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

    def _encode(self, state):
        return self._encoder[state]

    def _decode(self, i):
        return self._decoder[i]
