import numpy as np
import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

class DQN:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.memory = []
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.state_space))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
    def experience_replay(self, batch_size):
        """Store and sample experiences from the environment."""
        # Store the most recent experiences in a replay buffer
        for state, action, reward, next_state, done in self.memory:
            self.replay_buffer.append((state, action, reward, next_state, done))

        # Sample a random batch of experiences from the replay buffer
        batch = self.replay_buffer.sample(batch_size)

        # Update the Q-network using the sampled experiences
        self.update_q_network(batch)

    def update_exploration_rate(self):
        self.epsilon = self.epsilon * self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min


    # ... (rest of the code remains the same)


# Create the environment
env = gym.make('MsPacman-v0')

state_space = env.observation_space.shape
action_space = env.action_space.n

# Create the DQN agent
agent = DQN(state_space, action_space)

# Training loop
batch_size = 64
episodes = 100

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        # Preprocess state (if necessary) and reshape it
        state = np.expand_dims(state, axis=0)

        # Choose action
        action = agent.act(state)

        # Take the action in the environment
        step_result = env.step(action)
        next_state = step_result[0]
        reward = step_result[1]
        done = step_result[2]


        # Preprocess next_state (if necessary) and reshape it
        next_state = np.expand_dims(next_state, axis=0)

        # Store the experience in memory
        agent.remember(state, action, reward, next_state, done)

        # Update the state
        state = next_state
        total_reward += reward

    # Perform experience replay
    #agent.experience_replay(batch_size)

    # Update exploration rate
    #agent.update_exploration_rate()

    # Print episode statistics
    print(f"Episode: {episode+1}, Total Reward: {total_reward}")

# Save the trained model
agent.model.save('pacman_dqn_model.h5')
