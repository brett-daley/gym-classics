import gym
import gym_classics
gym_classics.register('gym')
import numpy as np

# Hyperparameters for Q-Learning
discount = 0.9
epsilon = 0.5
learning_rate = 0.025

# Instantiate the environment
env = gym.make('ClassicGridworld-v0')
state, _ = env.reset(seed=0)

# Our Q-function is a numpy array
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Loop for 500k timesteps
for _ in range(500000):
    # Select action from ε-greedy policy
    if env.np_random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])

    # Step the environment
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    # Q-Learning update:
    # Q(s,a) <-- Q(s,a) + α * (r + γ max_a' Q(s',a') - Q(s,a))
    td_error = reward - Q[state, action]
    if not done:
        td_error += discount * np.max(Q[next_state])
    Q[state, action] += learning_rate * td_error

    # Reset the environment if we're done
    if done:
        next_state, _ = env.reset()
    state = next_state

# Now let's see what the value function looks like after training:
V = np.max(Q, axis=1)
print(V)
