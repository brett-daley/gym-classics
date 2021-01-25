from gym_classics.envs.abstract.base_env import BaseEnv


class Gridworld(BaseEnv):
    """Abstract class for creating gridworld-type environments."""

    def __init__(self, dims, starts, blocks=frozenset(), n_actions=None):
        self._dims = dims
        self._blocks = blocks

        if n_actions is None:
            n_actions = 4
        super().__init__(starts, n_actions)

    def _next_state(self, state, action):
        next_state = self._move(state, action)
        if self._is_blocked(next_state):
            next_state = state
        return self._clamp(next_state), 1.0

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
        yield self._deterministic_step(state, action)
