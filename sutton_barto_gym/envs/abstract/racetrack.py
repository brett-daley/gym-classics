from gym.spaces import Discrete
import numpy as np

from sutton_barto_gym.envs.abstract.gridworld import Gridworld


class Racetrack(Gridworld):
    """Abstract class for creating racetrack environments.

    See page 112 of Sutton & Barto (2018).
    """

    def __init__(self, track):
        """Instantiates a racetrack environment.

        Args:
            track
        """
        self._states = self._get_coordinates(track, value=0)
        blocks = self._get_coordinates(track, value=1)
        self._start_line = self._get_coordinates(track, value=2)
        self._start_line_tuple = list(self._start_line)  # For efficient sampling
        self._finish_line = self._get_coordinates(track, value=3)

        self._max_velocity = 5  # Each velocity component must be in [0, 5]

        super().__init__(dims=track.shape[::-1], start=None, blocks=blocks)

        # Our state is a tuple of position and velocity, both of which are
        # themselves 2-tuples representing their x- and y-components
        self._state = None

        # There are 9 actions: both velocity components can be changed by {-1,0,+1}
        self.action_space = Discrete(9)
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

    def _get_coordinates(self, track, value):
        Y, X = np.where(track == value)
        Y = [track.shape[0] - 1 - y for y in Y]
        return frozenset(zip(X, Y))

    def reset(self):
        i = self.np_random.choice(len(self._start_line_tuple))
        pos = self._start_line_tuple[i]
        vel = (0, 0)
        self._state = (pos, vel)
        return self._encode(self._state)

    def _decoded_states(self):
        for pos_x in range(self._dims[0]):
            for pos_y in range(self._dims[1]):
                pos = (pos_x, pos_y)
                if not self._reachable(pos):
                    continue

                for vel_x in range(self._max_velocity + 1):
                    for vel_y in range(self._max_velocity + 1):
                        vel = (vel_x, vel_y)
                        yield (pos, vel)

    def _next_state(self, state, action):
        # Only 90% chance that we modify the velocity
        update_vel = (self.np_random.rand() < 0.9)
        next_state = self._move(state, action, update_vel)

        # If we need to terminate, just point back to the current state
        # because we know it is guaranteed to exist.
        if self._done(state, action, next_state):
            next_state = state
        return next_state

    def _move(self, state, action, update_vel):
        ((pos_x, pos_y), (vel_x, vel_y)) = state

        if update_vel:
            # Update velocity
            delta_vel_x, delta_vel_y = self._action_decoder[action]
            vel_x = np.clip(vel_x + delta_vel_x, 0, self._max_velocity)
            vel_y = np.clip(vel_y + delta_vel_y, 0, self._max_velocity)

        # Update position
        pos_x += vel_x
        pos_y += vel_y

        position = (pos_x, pos_y)
        velocity = (vel_x, vel_y)
        return (position, velocity)

    def _reachable(self, position):
        return not (self._out_of_bounds(position) or (position in self._finish_line))

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
        return not self._reachable(next_pos)

    def _reward(self, state, action, next_state):
        next_pos, _ = next_state
        if self._out_of_bounds(next_pos):
            return -1.0
        elif next_pos in self._finish_line:
            return 1.0
        else:
            return 0.0

    def _generate_transitions(self, state, action):
        state = self._decode(state)

        for update_vel in [False, True]:
            next_state = self._move(state, action, update_vel)
            reward = self._reward(state, action, next_state)
            prob = (0.9 if update_vel else 0.1)
            done = self._done(state, action, next_state)

            # If we need to terminate, just point back to the current state
            # because we know it is guaranteed to exist.
            if done:
                next_state = state
            yield self._encode(next_state), reward, prob, done
