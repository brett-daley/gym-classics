import gym
import gym_classics
import numpy as np

# Hyperparameters for Q-Learning
discount = 0.9
epsilon = 0.5
learning_rate = 0.025

# Instantiate the environment
env = gym.make('ClassicGridworld-v0')
state = env.reset()

# Set seeds for reproducibility
np.random.seed(0)
env.seed(0)

# Our Q-function is a numpy array
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Loop for 500k timesteps
for _ in range(500000):
    # Select action from ε-greedy policy
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])

    # Step the environment
    next_state, reward, done, _ = env.step(action)

    # Q-Learning update:
    # Q(s,a) <-- Q(s,a) + α * (r + γ max_a' Q(s',a') - Q(s,a))
    target = reward - Q[state, action]
    if not done:
        target += discount * np.max(Q[next_state])
    Q[state, action] += learning_rate * target

    # Reset the environment if we're done
    state = env.reset() if done else next_state

# Now let's see what the value function looks like after training:
V = np.max(Q, axis=1)
print(V)
