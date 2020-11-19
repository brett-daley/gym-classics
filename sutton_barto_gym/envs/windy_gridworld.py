from sutton_barto_gym.building_blocks import AbstractGridworld


class WindyGridworld(AbstractGridworld):
    """Windy Gridworld.

    Page 130 of Sutton & Barto (2018).
    """

    def __init__(self):
        super().__init__(dims=(10, 7), start=(0, 3), goals={(7, 3)})

    def step(self, action):
        assert self.action_space.contains(action)
        state = self._apply_move(self._state, action)
        state = self._apply_wind(state)

        self._state = state
        reward = 0.0 if self._is_goal(state) else -1.0
        done = self._is_goal(state)
        return self._encode(state), reward, done, {}

    def _apply_wind(self, state):
        x, y = state
        state = (x, y + self._wind_strength())
        return self._clamp(state)

    def _wind_strength(self):
        """Returns wind strength in the current state."""
        x, _ = self._state
        if x in {3, 4, 5, 8}:
            return 1
        elif x in {6, 7}:
            return 2
        else:
            return 0
