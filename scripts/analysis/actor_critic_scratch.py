import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from gymnasium import spaces
import pandas as pd
import random

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


data = pd.read_csv("./data/crypto_exchange_rates.csv")

SEQUENCE_INCREMENT_MINUTES = 5
SEQUENCE_LENGTH_MINUTES = 5 * 24 * 60 # 5 days
EPISODE_LENGTH_MINUTES = 24 * 60

coin_values = data['pangolin'].values

# sequences = []
# for i in range(len(coin_values) - SEQUENCE_LENGTH_MINUTES):
#     idx = list(range(i, i + SEQUENCE_LENGTH_MINUTES, SEQUENCE_INCREMENT_MINUTES))
#     sequences.append(coin_values[idx])

# random.choice(sequences)

# Crypto Data:
# - Change this to sin-deltas
# - Price deltas
# - Hourly prices for 1 week
# - 5-minutes for last 12 hours
class SineWaveEnv(gym.Env):
    def __init__(self, step_size=0.4):
        super(SineWaveEnv, self).__init__()
        self.trade_cost = 0.99
        self.step_size = step_size
        self.current_step = 0
        self.action_space = spaces.Discrete(3)  # 0: SELL, 1: BUY, 2: HOLD
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.step_count = 0

        self.holdings_usd = 100
        self.holdings_coin = 0

    def step(self, action, max_steps):
        self.current_step += 1
        self.step_count += 1


        current_sequence = self.past_values(self.current_step)
        next_sequence = self.past_values(self.current_step + 1)

        current_value = current_sequence[-1]
        next_value = next_sequence[-1]

        current_holdings = self.total_holdings(current_value)

        if action == 1:
            self.buy(current_value)
        elif action == 0:
            self.sell(current_value)
        elif action == 2:
            pass

        next_holdings = self.total_holdings(next_value)
        delta_holdings = next_holdings - current_holdings

        if delta_holdings > 1000:
            import pdb; pdb.set_trace()

        reward = delta_holdings
        if self.step_count >= max_steps:
            if next_holdings == 100.0:
                reward = -1000.0
            else:
                reward = delta_holdings

        # print(f"Current Holdings: {current_holdings}, Next: {next_holdings}, Delta: {delta_holdings}, Reward: {reward}, Delta Value: {next_value - current_value}")

        if self.step_count >= max_steps:
            done = True
        else:
            done = False

        info = {
            "step_count": self.step_count,
        }

        return self.past_values(self.current_step), reward, done, info

    def buy(self, coin_value):
        # print(" - Buy")
        # print(f"Buying at {coin_value}. Current USD: {self.holdings_usd}, Coin: {self.holdings_coin}")
        self.holdings_coin += self.trade_cost * (self.holdings_usd / coin_value)
        self.holdings_usd = 0
        #print(f"Bought at {coin_value}. New USD: {self.holdings_usd}, Coin: {self.holdings_coin}")

    def sell(self, coin_value):
        # print(" - Sell")
        #print(f"Selling at {coin_value}. Current USD: {self.holdings_usd}, Coin: {self.holdings_coin}")
        self.holdings_usd += self.trade_cost * self.holdings_coin * coin_value
        self.holdings_coin = 0
        #print(f"Sold at {coin_value}. New USD: {self.holdings_usd}, Coin: {self.holdings_coin}")

    def total_holdings(self, coin_value):
        return self.holdings_usd + coin_value * self.holdings_coin

    def reset(self):
        self.step_count = 0
        self.holdings_usd = 100
        self.holdings_coin = 0

        # self.current_step = np.random.uniform(0, 2 * np.pi)
        self.current_step = random.randrange(0,len(coin_values) - SEQUENCE_LENGTH_MINUTES - EPISODE_LENGTH_MINUTES)
        return self.past_values(self.current_step)

    def past_values(self, init_step):
        idx = list(range(init_step, init_step + SEQUENCE_LENGTH_MINUTES, SEQUENCE_INCREMENT_MINUTES))
        return np.array(coin_values[idx])

def build_actor(state_shape, action_space):
    model = tf.keras.Sequential([
        layers.Input(shape=state_shape),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(action_space, activation='softmax')
    ])
    return model

def build_critic(state_shape):
    model = tf.keras.Sequential([
        layers.Input(shape=state_shape),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(1)
    ])
    return model

# Assuming your state is a sequence of states, update the shape accordingly
# For example, if you consider the last 5 states, the shape would be (5, 3)
with tf.device('/GPU:0'):
    actor = build_actor(state_shape=(EPISODE_LENGTH_MINUTES,), action_space=3)
    critic = build_critic(state_shape=(EPISODE_LENGTH_MINUTES,))

import tensorflow as tf

# Hyperparameters
num_episodes = 1000
learning_rate = 0.001
actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate)

# Initialize the environment
env = SineWaveEnv()

gamma = 0.99  # Discount factor
last_episode_reward = None

for episode in range(num_episodes):
    actions = []
    state = env.reset()
    episode_reward = 0
    done = False
    episode_states, episode_actions, episode_rewards = [], [], []

    while not done:
        action_probs = actor(np.array([state]), training=True)
        critic_value = critic(np.array([state]), training=True)

        action = np.random.choice(3, p=np.squeeze(action_probs))
        if random.random() > 0.95 and last_episode_reward is not None and last_episode_reward > -10 and last_episode_reward < 10:
            action = random.randint(0,3)

        actions.append(action)

        next_state, reward, done, info = env.step(action, EPISODE_LENGTH_MINUTES)

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
    discounted_rewards = tf.cast(tf.convert_to_tensor(discounted_rewards), dtype=tf.float32)

    with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
      # Recompute the probabilities and critic value
      current_probs = actor(episode_states, training=True)
      current_values = critic(episode_states, training=True)

      # Extract the probabilities of the chosen actions
      action_probs = tf.reduce_sum(current_probs * tf.one_hot(episode_actions, 3), axis=1)

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

    last_episode_reward = episode_reward

    print(f"Episode: {episode + 1}, Reward: {episode_reward}, Steps: {info['step_count']}, Actor Loss: {actor_loss}, Critic Loss: {critic_loss}")
    print("".join(map(str,actions)))
