from gym_classics.envs.abstract.gridworld import Gridworld


class NoisyGridworld(Gridworld):
    """Abstract class for creating gridworld-type environments with the classic
    80-10-10 stochastic dynamics:

        - 80% chance: action succeeds
        - 10% chance: action is rotated counter-clockwise
        - 10% chance: action is rotated clockwise
    """

    def _sample_random_elements(self, state, action):
        return [self._noisy_action(action)]

    def _next_state(self, state, action, noisy_action):
        next_state, _ = super()._next_state(state, noisy_action)
        if action == noisy_action:
            return next_state, 0.8
        return next_state, 0.1

    def _noisy_action(self, action):
        p = self.np_random.rand()
        # 10% chance: rotate the action clockwise
        if 0.8 <= p < 0.9:
            action += 1
        # 10% chance: rotate the action counter-clockwise
        elif 0.9 <= p:
            action -= 1
        return action % self.action_space.n

    def _generate_transitions(self, state, action):
        for i in [-1, 0, 1]:
            noisy_action = (action + i) % self.action_space.n
            yield self._deterministic_step(state, action, noisy_action)
