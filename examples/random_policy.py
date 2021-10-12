import gym
import gym_classics

env = gym.make('ClassicGridworld-v0')
state = env.reset()
for t in range(1, 100 + 1):
    action = env.action_space.sample()  # Select a random action
    next_state, reward, done, info = env.step(action)
    print("t={}, state={}, action={}, reward={}, next_state={}, done={}".format(
        t, state, action, reward, next_state, done))
    state = next_state if not done else env.reset()
env.close()
