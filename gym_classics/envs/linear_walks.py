from gym_classics.envs.abstract.linear_walk import LinearWalk


class Walk5(LinearWalk):
    """A linear walk with 5 states. Reward is 0 on the left and +1 on the right.

    Page 125 of Sutton & Barto (2018).
    """

    def __init__(self):
        super().__init__(length=5, left_reward=0.0, right_reward=1.0)


class Walk19(LinearWalk):
    """A linear walk with 19 states. Reward is -1 on the left and +1 on the right.

    Page 145 of Sutton & Barto (2018).
    """

    def __init__(self):
        super().__init__(length=19, left_reward=-1.0, right_reward=1.0)
