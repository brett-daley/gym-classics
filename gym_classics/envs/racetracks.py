import numpy as np

from gym_classics.envs.abstract.racetrack import Racetrack


class Racetrack1(Racetrack):
    """A gridworld-type racetrack where a racecar must traverse a right turn and reach
    the finish line as quickly as possible. The agent begins at a random location on the
    starting line and can only directly control the velocity of the racecar (not its
    position). Each velocity component can never be negative nor greater than 4. If the
    car goes out of bounds, it is reset to a random location on the starting line without
    terminating the episode. *NOTE:* While the original version forbids both velocity
    components from being zero simultaneously, no such restriction is enforced in this
    implementation.

    **reference:** cite{3} (page 112, figure 5.5, left).

    **state:** Racecar position and velocity.

    **actions:** Changes to the racecar's *velocity* (not position) vector, where the x-
    and y- components can be independently modified by {-1, 0, +1} on each timestep.
    This gives a total of 9 actions.

    **rewards:** -1 on all transitions unless the finish line is reached.

    **termination:** Reaching the finish line.
    """

    layout = """
|XXX             G|
|XX              G|
|XX              G|
|X               G|
|                G|
|                G|
|          XXXXXXX|
|         XXXXXXXX|
|         XXXXXXXX|
|         XXXXXXXX|
|         XXXXXXXX|
|         XXXXXXXX|
|         XXXXXXXX|
|         XXXXXXXX|
|X        XXXXXXXX|
|X        XXXXXXXX|
|X        XXXXXXXX|
|X        XXXXXXXX|
|X        XXXXXXXX|
|X        XXXXXXXX|
|X        XXXXXXXX|
|X        XXXXXXXX|
|XX       XXXXXXXX|
|XX       XXXXXXXX|
|XX       XXXXXXXX|
|XX       XXXXXXXX|
|XX       XXXXXXXX|
|XX       XXXXXXXX|
|XX       XXXXXXXX|
|XXX      XXXXXXXX|
|XXX      XXXXXXXX|
|XXXSSSSSSXXXXXXXX|
"""

    def __init__(self):
        super().__init__(Racetrack1.layout)


class Racetrack2(Racetrack):
    """Same as `Racetrack1` but with a different track layout.

    **reference:** cite{3} (page 112, figure 5.5, right).
    """

    layout = """
|XXXXXXXXXXXXXXXX               G|
|XXXXXXXXXXXXX                  G|
|XXXXXXXXXXXX                   G|
|XXXXXXXXXXX                    G|
|XXXXXXXXXXX                    G|
|XXXXXXXXXXX                    G|
|XXXXXXXXXXX                    G|
|XXXXXXXXXXXX                   G|
|XXXXXXXXXXXXX                  G|
|XXXXXXXXXXXXXX                XX|
|XXXXXXXXXXXXXX             XXXXX|
|XXXXXXXXXXXXXX            XXXXXX|
|XXXXXXXXXXXXXX          XXXXXXXX|
|XXXXXXXXXXXXXX         XXXXXXXXX|
|XXXXXXXXXXXXX          XXXXXXXXX|
|XXXXXXXXXXXX           XXXXXXXXX|
|XXXXXXXXXXX            XXXXXXXXX|
|XXXXXXXXXX             XXXXXXXXX|
|XXXXXXXXX              XXXXXXXXX|
|XXXXXXXX               XXXXXXXXX|
|XXXXXXX                XXXXXXXXX|
|XXXXXX                 XXXXXXXXX|
|XXXXX                  XXXXXXXXX|
|XXXX                   XXXXXXXXX|
|XXX                    XXXXXXXXX|
|XX                     XXXXXXXXX|
|X                      XXXXXXXXX|
|                       XXXXXXXXX|
|                       XXXXXXXXX|
|SSSSSSSSSSSSSSSSSSSSSSSXXXXXXXXX|
"""

    def __init__(self):
        super().__init__(Racetrack2.layout)
