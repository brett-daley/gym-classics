from gym_classics.envs.abstract.linear_walk import LinearWalk


class Walk5(LinearWalk):
    """A 5-state deterministic linear walk. Ideal for implementing random walk
    experiments.

    **reference:** cite{3} (page 125).

    **state:** Discrete position {0, ..., 4} on the number line.

    **actions:** Move left/right.

    **rewards:** +1 for moving right in the extreme right state.

    **termination:** Moving right in the extreme right state or moving left in the
    extreme left state.
    """

    def __init__(self):
        super().__init__(length=5, left_reward=0.0, right_reward=1.0)


class Walk19(LinearWalk):
    """Same as `5Walk` but with 19 states and an additional -1 reward for moving left
    in the extreme left state.

    **reference:** cite{3} (page 145).
    """

    def __init__(self):
        super().__init__(length=19, left_reward=-1.0, right_reward=1.0)
