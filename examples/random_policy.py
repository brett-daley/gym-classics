import gym                    # or `import gymnasium as gym`
import gym_classics
gym_classics.register('gym')  # or `gym_classics.register('gymnasium')`

env = gym.make('ClassicGridworld-v0')
state, _ = env.reset(seed=0)

for t in range(1, 100 + 1):
    action = env.action_space.sample()  # Select a random action
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    print("t={}, state={}, action={}, reward={}, next_state={}, done={}".format(
        t, state, action, reward, next_state, done))
    if done:
        next_state, _ = env.reset()
    state = next_state

env.close()  # Optional, not currently implemented by any environments
