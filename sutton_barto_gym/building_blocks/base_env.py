import gym


class BaseEnv(gym.Env):
    """Abstract base class for shared functionality between all environments."""

    def seed(self, seed=None):
        self.action_space.seed(seed)
        return [seed]
