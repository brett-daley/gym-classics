from collections import deque

import numpy as np

from sutton_barto_gym.building_blocks import AbstractGridworld


class ClassicGridworld(AbstractGridworld):
    """'Classic' Gridworld.

    Norvig & Russel (2020, 4th ed.).
    """

    def __init__(self):
        super().__init__(dims=(4, 3), start=(0, 0), blocks={(1, 1)})

    def step(self, action):
        assert self.action_space.contains(action)
        state = self._state
        info = {}

        if state == (3, 1):
            return self._encode(state), -1.0, True, info
        elif state == (3, 2):
            return self._encode(state), 1.0, True, info

        action = self._noisy_action(action)
        state = self._apply_move(state, action)

        self._state = state
        return self._encode(state), 0.0, False, info

    def _noisy_action(self, action):
        x = np.random.rand()

        # 80% chance the action is unaffected
        if x < 0.8:
            return action

        # We use a deque for easy rotation
        all_actions = deque(self.actions())
        all_actions.rotate(-action)

        # 10% chance: rotate the action counter-clockwise
        if 0.8 <= x < 0.9:
            all_actions.rotate(1)
        # 10% chance: rotate the action clockwise
        else:
            all_actions.rotate(-1)

        return all_actions[action]
