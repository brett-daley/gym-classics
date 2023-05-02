import warnings


_registry = (
    {
        'id': '5Walk-v0',
        'entry_point': 'gym_classics.envs.linear_walks:Walk5'
    },
    {
        'id': '19Walk-v0',
        'entry_point': 'gym_classics.envs.linear_walks:Walk19'
    },
    {
        'id': 'ClassicGridworld-v0',
        'entry_point': 'gym_classics.envs.classic_gridworld:ClassicGridworld'
    },
    {
        'id': 'CliffWalk-v0',
        'entry_point': 'gym_classics.envs.cliff_walk:CliffWalk'
    },
    {
        'id': 'DynaMaze-v0',
        'entry_point': 'gym_classics.envs.dyna_maze:DynaMaze',
    },
    {
        'id': 'FourRooms-v0',
        'entry_point': 'gym_classics.envs.four_rooms:FourRooms',
    },
    {
        'id': 'JacksCarRental-v0',
        'entry_point': 'gym_classics.envs.jacks_car_rental:JacksCarRental',
        'max_episode_steps': 100
    },
    {
        'id': 'JacksCarRentalModified-v0',
        'entry_point': 'gym_classics.envs.jacks_car_rental:JacksCarRentalModified',
        'max_episode_steps': 100,
    },
    # {
    #     'id': 'Racetrack1-v0',
    #     'entry_point': 'gym_classics.envs.racetracks:Racetrack1',
    # },
    # {
    #     'id': 'Racetrack2-v0',
    #     'entry_point': 'gym_classics.envs.racetracks:Racetrack2',
    # },
    {
        'id': 'SparseGridworld-v0',
        'entry_point': 'gym_classics.envs.sparse_gridworld:SparseGridworld',
    },
    {
        'id': 'WindyGridworld-v0',
        'entry_point': 'gym_classics.envs.windy_gridworld:WindyGridworld',
    },
    {
        'id': 'WindyGridworldKings-v0',
        'entry_point': 'gym_classics.envs.windy_gridworld:WindyGridworldKings',
    },
    {
        'id': 'WindyGridworldKingsNoOp-v0',
        'entry_point': 'gym_classics.envs.windy_gridworld:WindyGridworldKingsNoOp',
    },
    {
        'id': 'WindyGridworldKingsStochastic-v0',
        'entry_point': 'gym_classics.envs.windy_gridworld:WindyGridworldKingsStochastic',
    }
)


_backend = None

def register(backend='gym'):
    global _backend
    if _backend is not None:
        warnings.warn("gym-classics environments were already registered for {}; "
                      "additional calls to `register()` are ignored.".format(_backend))
        return

    assert backend in {'gym', 'gymnasium'}
    _backend = backend

    if backend == 'gym':
        import gym
        register = gym.envs.register
    elif backend == 'gymnasium':
        import gymnasium
        register = gymnasium.register

    for kwargs in _registry:
        register(**kwargs)
