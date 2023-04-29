from gym_classics.envs.abstract.base_env import BaseEnv


class Gridworld(BaseEnv):
    """Abstract class for creating gridworld-type environments."""

    def __init__(self, layout_string, n_actions=None):
        self.dims, starts, self._goals, self._blocks = parse_gridworld(layout_string)

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
        x = max(0, min(x, self.dims[0] - 1))
        y = max(0, min(y, self.dims[1] - 1))
        return (x, y)

    def _is_blocked(self, state):
        """Returns True if this state cannot be occupied, False otherwise."""
        return state in self._blocks

    def _generate_transitions(self, state, action):
        yield self._deterministic_step(state, action)


def parse_gridworld(layout_string):
    layout_string = layout_string.replace('|', '')  # Remove optional pipe characters
    lines = layout_string.split('\n')
    lines = [l for l in lines if l != '']  # Remove empty lines

    # Get dimensions: assume rectangular (width, height)
    H = len(lines)
    W = len(lines[0])
    for l in lines:
        assert len(l) == W, "layout string is not rectangular; check dimensions"
    dims = (W, H)

    starts = set()
    goals = set()
    blocks = set()

    for row in range(H):
        for col in range(W):
            coords = (col, H - 1 - row)  # Makes (0,0) the bottom-left cell in the gridworld
            char = lines[row][col]

            if char == 'S':  # Start (may be more than one)
                starts.add(coords)
            elif char == 'G':  # Goal (may be more than one)
                goals.add(coords)
            elif char == 'X':  # Block (agent cannot occupy these cells)
                blocks.add(coords)
            elif char == ' ':  # Empty (agent can occupy these cells)
                pass
            else:
                raise ValueError(f"invalid character '{char}' at {coords}")

    return dims, frozenset(starts), frozenset(goals), frozenset(blocks)
