import numpy as np

from gym_classics.envs.abstract.gridworld import Gridworld
from gym_classics.utils import clip


class Racetrack(Gridworld):
    """Abstract class for creating racetrack environments.

    See page 112 of Sutton & Barto (2018).
    """

    def __init__(self, track):
        """Instantiates a racetrack environment.

        State is a tuple of position and velocity, both of which are themselves 2-tuples
        representing their x- and y-components.

        Args:
            track
        """
        blocks = self._get_coordinates(track, value=1)
        starting_line = self._get_coordinates(track, value=2)
        self._finish_line = self._get_coordinates(track, value=3)

        self._max_velocity = 5  # Each velocity component must be in [0, 5]

        # There are 9 actions: both velocity components can be changed by {-1,0,+1}
        self._action_decoder = {
            0: (-1, -1),
            1: (-1,  0),
            2: (-1, +1),
            3: ( 0, -1),
            4: ( 0,  0),
            5: ( 0, +1),
            6: (+1, -1),
            7: (+1,  0),
            8: (+1, +1),
        }

        starts = {(pos, (0, 0)) for pos in starting_line}
        super().__init__(dims=track.shape[::-1], starts=starts, blocks=blocks, n_actions=9)

    def _get_coordinates(self, track, value):
        Y, X = np.where(track == value)
        Y = [track.shape[0] - 1 - y for y in Y]
        return frozenset(zip(X, Y))

    def _sample_random_elements(self, state, action):
        # Only 90% chance that the velocity is successfully modified
        success = (self.np_random.rand() < 0.9)
        return [success]

    def _next_state(self, state, action, success):
        ((pos_x, pos_y), (vel_x, vel_y)) = state

        if success:
            # Update velocity
            delta_vel_x, delta_vel_y = self._action_decoder[action]
            vel_x = clip(vel_x + delta_vel_x, 0, self._max_velocity)
            vel_y = clip(vel_y + delta_vel_y, 0, self._max_velocity)

        # Update position
        pos_x += vel_x
        pos_y += vel_y

        position = (pos_x, pos_y)
        velocity = (vel_x, vel_y)
        state = (position, velocity)

        prob = 0.9 if success else 0.1
        return state, prob

    def _out_of_bounds(self, position):
        x, y = position
        if not (0 <= x < self._dims[0]):
            return True
        if not (0 <= y < self._dims[1]):
            return True
        if position in self._blocks:
            return True
        return False

    def _done(self, state, action, next_state):
        next_pos, _ = next_state
        return self._out_of_bounds(next_pos) or next_pos in self._finish_line

    def _reward(self, state, action, next_state):
        next_pos, _ = next_state
        if self._out_of_bounds(next_pos):
            return -1.0
        elif next_pos in self._finish_line:
            return 1.0
        else:
            return 0.0

    def _generate_transitions(self, state, action):
        for success in [False, True]:
            yield self._deterministic_step(state, action, success)
