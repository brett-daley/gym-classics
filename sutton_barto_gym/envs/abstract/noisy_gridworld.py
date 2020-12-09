from collections import deque

import numpy as np

from sutton_barto_gym.envs.abstract.gridworld import Gridworld


class NoisyGridworld(Gridworld):
    """Abstract class for creating gridworld-type environments with the classic
    80-10-10 stochastic dynamics:

        - 80% chance: action succeeds
        - 10% chance: action is rotated counter-clockwise
        - 10% chance: action is rotated clockwise
    """

    def _apply_move(self, state, action):
        action = self._noisy_action(action)
        return super()._apply_move(state, action)

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

    def _generate_transitions(self, state, action):
        state = self._decode(state)

        all_actions = deque(self.actions())
        all_actions.rotate(-action)

        for i in [-1, 0, 1]:
            a = all_actions[i]
            next_state = super()._next_state(state, a)
            reward = self._reward(state, a, next_state)
            prob = (0.8 if i == 0 else 0.1)
            done = self._done(state, a, next_state)
            yield self._encode(next_state), reward, prob, done
