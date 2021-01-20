
def clip(x, low, high):
    """A scalar version of numpy.clip. Much faster because it avoids memory allocation."""
    return min(max(x, low), high)


def print_gridworld(env, array, decimals=2, separator=' ' * 2, signed=True, transpose=False):
    # First get the string length of the longest number
    def formatter(x):
        string = '{:' + ('+' if signed else '') + '.' + str(decimals) + 'f}'
        return string.format(x)
    maxlen = max([len(formatter(x)) for x in array])

    # Now we can actually print the values
    for y in reversed(range(env._dims[1])):
        for x in range(env._dims[0]):
            state = (x, y) if not transpose else (y, x)
            if env._is_reachable(state):
                s = env._encode(state)
                print(formatter(array[s]).rjust(maxlen), end=separator)
            else:
                print(' ' * maxlen, end=separator)
        print()


def print_racetrack(env, V):
    grid_values = np.zeros(env._dims)

    reachable_positions = [pos for (pos, _) in env._reachable_states]
    velocities = [(x, y) for x in range(env._max_velocity + 1)
                            for y in range(env._max_velocity + 1)]

    # Helper function to get reachable velocity pairings in this position
    def valid_states_in_position(pos):
        states = [(pos, vel) for vel in velocities]
        return np.asarray([env._encode(s) for s in states if s in env._reachable_states])

    # Set each cell to be the maximum value over the velocities
    for x in range(env._dims[0]):
        for y in range(env._dims[1]):
            pos = (x, y)
            if pos in reachable_positions:
                states = valid_states_in_position(pos)
                grid_values[x, y] = np.max(V[states])
            else:
                grid_values[x, y] = np.nan

    # First get the string length of the longest number
    formatter = lambda v: '{:+.2f}'.format(v)
    maxlen = max([len(formatter(v)) for v in grid_values.flatten()])

    # Now we can actually print the values
    for y in reversed(range(env._dims[1])):
        for x in range(env._dims[0]):
            v = grid_values[x,y]
            if not np.isnan(v):
                print(formatter(v).rjust(maxlen), end=' ')
            else:
                print(' ' * maxlen, end=' ')
        print()
