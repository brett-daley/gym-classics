from gym_classics.envs.abstract.gridworld import Gridworld


class WindyGridworld(Gridworld):
    """A 10x7 deterministic gridworld where some columns are affected by an upward wind.
    The agent starts in cell (0, 3) and the goal is at cell (7, 3). If an agent executes
    an action from a cell with wind, the resulting position is given by the vector sum
    of the action's effect and the wind.

    **reference:** cite{3} (page 130, example 6.5).

    **state:** Grid location.

    **actions:** Move up/right/down/left.

    **rewards:** -1 for all transitions unless the episode terminates.

    **termination:** Reaching the goal.
    """

    def __init__(self):
        self._goal = (7, 3)
        super().__init__(dims=(10, 7), starts={(0, 3)})

    def _next_state(self, state, action):
        wind_strength = self._wind_strength(state)
        state, _ = super()._next_state(state, action)
        state = self._apply_wind(state, wind_strength)
        return self._clamp(state), 1.0

    def _apply_wind(self, state, strength):
        x, y = state
        return (x, y + strength)

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
        return 0.0 if self._done(state, action, next_state) else -1.0

    def _done(self, state, action, next_state):
        return next_state == self._goal


class WindyGridworldKings(WindyGridworld):
    """Same as `WindyGridworld` but with diagonal "King's" moves permitted.

    **reference:** cite{3} (page 131, exercise 6.9).

    **actions:** Move in the 4 cardinal directions and 4 intermediate directions.
    """

    def __init__(self):
        self._goal = (7, 3)
        super(WindyGridworld, self).__init__(dims=(10, 7), starts={(0, 3)}, n_actions=8)

    def _move(self, state, action):
        if action < 4:
            # Standard cardinal direction
            return super()._move(state, action)

        x, y = state
        return {
            4: (x+1, y+1),  # Up-Right
            5: (x+1, y-1),  # Down-Right
            6: (x-1, y-1),  # Down-Left
            7: (x-1, y+1)   # Up-Left
        }[action]


class WindyGridworldKingsNoOp(WindyGridworldKings):
    """Same as `WindyGridworldKings` but with an extra "no-op" (do nothing) action.

    **reference:** cite{3} (page 131, exercise 6.9).

    **actions:** Move in the 8 cardinal/intermediate directions or take a no-op action.
    """

    def __init__(self):
        self._goal = (7, 3)
        super(WindyGridworld, self).__init__(dims=(10, 7), starts={(0, 3)}, n_actions=9)

    def _move(self, state, action):
        if action == 8:  # No-op
            return state
        return super()._move(state, action)


class WindyGridworldKingsStochastic(WindyGridworldKings):
    """Same as `WindyGridworldKings` but windy cells exhibit stochastic behavior:
    -1, +0, or +1 wind strength with probability 1/3 each.

    **reference:** cite{3} (page 131, exercise 6.10).
    """

    def _sample_random_elements(self, state, action):
        # 1/3 chance each: decreased, unchanged, or increased wind strength
        wind_delta = self.np_random.choice([-1, 0, 1])
        return [wind_delta]

    def _next_state(self, state, action, wind_delta):
        wind_strength = super()._wind_strength(state)
        state, _ = super()._next_state(state, action)
        if wind_strength > 0:
            wind_strength += wind_delta
            state = self._apply_wind(state, wind_strength)
            prob = 1/3
        else:
            prob = 1.0
        return self._clamp(state), prob

    def _generate_transitions(self, state, action):
        if self._wind_strength(state) == 0:
            yield self._deterministic_step(state, action, 0)
            return

        for wind_delta in [-1, 0, 1]:
            yield self._deterministic_step(state, action, wind_delta)
