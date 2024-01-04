import gymnasium as gym
import numpy as np
from gymnasium import spaces
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class SineWaveEnv(gym.Env):
    def __init__(self, step_size=0.4):
        super(SineWaveEnv, self).__init__()
        self.history_size = 10

        self.step_size = step_size
        self.current_step = 0
        self.action_space = spaces.Discrete(2)  # 0: DOWN, 1: UP
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.step_count = 0

    def step(self, action, max_steps):
        self.current_step += self.step_size
        self.step_count += 1

        next_value = np.sin(self.current_step)
        current_value = np.sin(self.current_step - self.step_size)

        # Determine reward
        if (next_value > current_value and action == 1) or (next_value <= current_value and action == 0):
            reward = 1
        else:
            reward = -1

        self.cumulative_reward += reward

        if self.step_count >= max_steps:
            done = True
        else:
            done = False

        info = {
            "step_count": self.step_count,
            "cumulative_reward": self.cumulative_reward
        }  # Additional info, not necessary in this case

        return self.past_values(self.current_step), reward, done, info

    def reset(self):
        self.step_count = 0
        self.cumulative_reward = 0

        self.current_step = np.random.uniform(0, 2 * np.pi)
        return self.past_values(self.current_step)

    def past_values(self, init_step):
        history = np.array([])
        for i in reversed(range(self.history_size)):
            sinval = np.sin(init_step - self.step_size * i)
            history = np.append(history, sinval)
        return history
            

import tensorflow as tf
from tensorflow.keras import layers

def build_actor(state_shape, action_space):
    model = tf.keras.Sequential([
        layers.Input(shape=state_shape),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(action_space, activation='softmax')
    ])
    return model

def build_critic(state_shape):
    model = tf.keras.Sequential([
        layers.Input(shape=state_shape),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])
    return model

# actor = build_actor(state_shape=(3,), action_space=2)
# critic = build_critic(state_shape=(3,))

# def build_actor(state_shape, action_space):
#     model = tf.keras.Sequential([
#         layers.Input(shape=state_shape),
#         layers.LSTM(128, return_sequences=True),
#         layers.LSTM(128),
#         layers.Dense(action_space, activation='softmax')
#     ])
#     return model

# def build_critic(state_shape):
#     model = tf.keras.Sequential([
#         layers.Input(shape=state_shape),
#         layers.LSTM(128, return_sequences=True),
#         layers.LSTM(128),
#         layers.Dense(1)
#     ])
#     return model

# Assuming your state is a sequence of states, update the shape accordingly
# For example, if you consider the last 5 states, the shape would be (5, 2)
with tf.device('/GPU:0'):
    actor = build_actor(state_shape=(10,), action_space=2)
    critic = build_critic(state_shape=(10,))

import tensorflow as tf

# Hyperparameters
num_episodes = 1000
learning_rate = 0.01
actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate)

# Initialize the environment
env = SineWaveEnv()

longest_run = 20

gamma = 1.0  # Discount factor

for episode in range(num_episodes):
    actions = []
    state = env.reset()
    episode_reward = 0
    done = False
    episode_states, episode_actions, episode_rewards = [], [], []

    while not done:
        action_probs = actor(np.array([state]), training=True)
        critic_value = critic(np.array([state]), training=True)

        action = np.random.choice(2, p=np.squeeze(action_probs))
        actions.append(action)

        next_state, reward, done, info = env.step(action, longest_run)

        # Store states, actions, and rewards
        episode_states.append(state)
        episode_actions.append(action)
        episode_rewards.append(reward)

        state = next_state
        episode_reward += reward

    # Calculate discounted rewards
    discounted_rewards = []
    cumulative_reward = 0
    for reward in episode_rewards[::-1]:
        cumulative_reward = reward + gamma * cumulative_reward
        discounted_rewards.insert(0, cumulative_reward)

    # Convert lists to tensor
    episode_states = tf.convert_to_tensor(episode_states)
    episode_actions = tf.convert_to_tensor(episode_actions)
    discounted_rewards = tf.convert_to_tensor(discounted_rewards)

    with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
      # Recompute the probabilities and critic value
      current_probs = actor(episode_states, training=True)
      current_values = critic(episode_states, training=True)

      # Extract the probabilities of the chosen actions
      action_probs = tf.reduce_sum(current_probs * tf.one_hot(episode_actions, 2), axis=1)

      # Calculate actor loss
      advantages = tf.squeeze(discounted_rewards) - tf.squeeze(current_values)
      actor_loss = -tf.reduce_mean(tf.math.log(action_probs + 1e-10) * tf.stop_gradient(advantages))

      # Calculate critic loss
      critic_loss = tf.keras.losses.MSE(tf.squeeze(discounted_rewards), tf.squeeze(current_values))

    # Backpropagation
    actor_grads = actor_tape.gradient(actor_loss, actor.trainable_variables)
    critic_grads = critic_tape.gradient(critic_loss, critic.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
    critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

    print(f"Episode: {episode + 1}, Reward: {episode_reward}, Steps: {info['step_count']}, Longest Run: {longest_run}, Actor Loss: {actor_loss}, Critic Loss: {critic_loss}")
    print("".join(map(str,actions)))
