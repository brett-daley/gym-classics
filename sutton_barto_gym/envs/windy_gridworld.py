from sutton_barto_gym.envs.abstract.gridworld import Gridworld


class WindyGridworld(Gridworld):
    """Windy Gridworld.

    Page 130 of Sutton & Barto (2018).
    """

    def __init__(self):
        super().__init__(dims=(10, 7), start=(0, 3), goals={(7, 3)})

    def _apply_move(self, state, action):
        wind_strength = self._wind_strength(state)
        state = super()._apply_move(state, action)
        return self._apply_wind(state, wind_strength)

    def _apply_wind(self, state, strength):
        x, y = state
        state = (x, y + strength)
        return self._clamp(state)

    def _wind_strength(self, state):
        """Returns wind strength in the given state."""
        x, _ = state
        if x in {3, 4, 5, 8}:
            return 1
        elif x in {6, 7}:
            return 2
        else:
            return 0

    def _reward(self, state, action, next_state):
        return 0.0 if self._is_goal(next_state) else -1.0

    def _done(self, state, action, next_state):
        return self._is_goal(next_state)
