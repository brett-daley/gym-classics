import numpy as np

from gym_classics.envs.abstract.gridworld import Gridworld
from gym_classics.utils import clip

class Racetrack(Gridworld):
    """Abstract class for creating racetrack environments."""

    def __init__(self, track):
        self.track = track
        blocks = self._get_coordinates(track, value=1)
        self._starting_line = self._get_coordinates(track, value=2)
        self._finish_line = self._get_coordinates(track, value=3)

        self._max_velocity = 4  # Each velocity component must be in [0, 5)

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

        starts = {(pos, (0, 0)) for pos in self._starting_line}
        super().__init__(dims=track.shape[::-1], starts=starts, blocks=blocks, n_actions=9)

    def _get_coordinates(self, track, value):
        Y, X = np.where(track == value)
        Y = [track.shape[0] - 1 - y for y in Y]
        return frozenset(zip(X, Y))

    def _sample_random_elements(self, state, action):
        # Only 90% chance that the velocity is successfully modified
        success = (self.np_random.rand() < 0.9)
        # Sample a random starting location in case we go out of bounds
        start_index = self.np_random.choice(len(self._starts))
        return [success, start_index]

    def _next_state(self, state, action, success, start_index):
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

        if self._out_of_bounds(position):
            # If we go out of bounds, we teleport to a random starting location
            position, velocity = self._starts[start_index]

        state = (position, velocity)

        # We must normalize the transition probability by the number of starting locations
        prob = (0.9 if success else 0.1) / len(self._starts)
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
        return next_pos in self._finish_line

    def _reward(self, state, action, next_state):
        return 0.0 if self._done(state, action, next_state) else -1.0

    def _generate_transitions(self, state, action):
        for success in [False, True]:
            for start_index in range(len(self._starts)):
                yield self._deterministic_step(state, action, success, start_index)

    def render(self, mode='human', scale=10):
        """ Set up render mode.
        @scale: How many times do you want to scale the visualization?
        """
        if not hasattr(self, "pygame"):
            try:
                import pygame
            except Exception as e:
                print("Please install pygame to see the visualization. You can use this command line:\npip install pygame\n")
                raise e
            self.scale = scale
            self.pygame = pygame
            pygame.init()
            self.display = pygame.display.set_mode(self._window_shape())

    def step(self, action):
        if hasattr(self, "pygame"):
            x,y = self._state[0]
            vis = self.track.copy().T
            vis[x,-1-y] = 9 # highlight the current pos
            vis = 255*vis/vis.max()
            surf = self.pygame.surfarray.make_surface(vis)
            surf = self.pygame.transform.scale(surf, self._window_shape())
            self.display.blit(surf, (0, 0))
            self.pygame.display.update()
        super().step(action)

    def close(self):
        if hasattr(self, "pygame"):
            self.pygame.quit()
        super().close()

    def _window_shape(self):
        """ Get properly scaled window shape."""
        return np.array(self.track.T.shape)*self.scale
