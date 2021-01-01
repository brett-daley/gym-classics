
def clip(x, low, high):
    """A scalar version of numpy.clip. Much faster because it avoids memory allocation."""
    return min(max(x, low), high)
