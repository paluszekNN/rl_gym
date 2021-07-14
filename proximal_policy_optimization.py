import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import scipy.signal
import time
from classic_control_envs import RLAlgorithm


def discounted_cumulative_sums(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


class Model:
    def __init__(self, input_shape, output_shpape, layers, activation_layers, squeeze, activation_output=None):
        input = keras.Input((input_shape,))
        x = input
        for layer in layers:
            x = keras.layers.Dense(layer, activation_layers)(x)
        output = keras.layers.Dense(output_shpape, activation_output)(x)
        if squeeze:
            output = tf.squeeze(output, axis=1)
        self.model = keras.Model(inputs=input, outputs=output)


class ProximalPolicyOptimization(RLAlgorithm):
    def __init__(self, gamma, steps_per_epoch, max_time, clip_ratio, policy_learning_rate, value_function_learning_rate,
                 train_policy_iterations, train_value_iterations, lam, target_kl, hidden_sizes, env):
        self.observation_dimensions = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.env = env

        self.steps_per_epoch = 4000
        self.max_time = max_time
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.train_policy_iterations = train_policy_iterations
        self.train_value_iterations = train_value_iterations
        self.lam = lam
        self.target_kl = target_kl

        self.buffer = Buffer(self.observation_dimensions, steps_per_epoch)

        self.actor = Model(self.observation_dimensions, self.num_actions, list(hidden_sizes), tf.tanh, False).model
        self.critic = Model(self.observation_dimensions, 1, list(hidden_sizes), tf.tanh, True).model

        self.policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
        self.value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)

    def action(self, env, obs):
        pass

    def logprobabilities(self, logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, self.num_actions) * logprobabilities_all, axis=1
        )
        return logprobability

    # Sample action from actor
    @tf.function
    def sample_action(self, observation):
        logits = self.actor(observation)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action

    # Train the policy by maxizing the PPO-Clip objective
    @tf.function
    def train_policy(self,
                     observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
                     ):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(
                self.logprobabilities(self.actor(observation_buffer), action_buffer)
                - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

        kl = tf.reduce_mean(
            logprobability_buffer
            - self.logprobabilities(self.actor(observation_buffer), action_buffer)
        )
        kl = tf.reduce_sum(kl)
        return kl

    # Train the value function by regression on mean-squared error
    @tf.function
    def train_value_function(self, observation_buffer, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean((return_buffer - self.critic(observation_buffer)) ** 2)
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))

    def train(self):
        observation, episode_return, episode_length = self.env.reset(), 0, 0
        time_start = time.time()

        while True:
            sum_return = 0
            sum_length = 0
            num_episodes = 0

            # Iterate over the steps of each epoch
            for t in range(self.steps_per_epoch):
                # Get the logits, action, and take one step in the environment
                observation = observation.reshape(1, -1)
                logits, action = self.sample_action(observation)
                observation_new, reward, done, _ = self.env.step(action[0].numpy())
                episode_return += reward
                episode_length += 1

                # Get the value and log-probability of the action
                value_t = self.critic(observation)
                logprobability_t = self.logprobabilities(logits, action)

                # Store obs, act, rew, v_t, logp_pi_t
                self.buffer.store(observation, action, reward, value_t, logprobability_t)

                # Update the observation
                observation = observation_new

                # Finish trajectory if reached to a terminal state
                terminal = done
                if terminal or (t == self.steps_per_epoch - 1):
                    last_value = 0 if done else self.critic(observation.reshape(1, -1))
                    self.buffer.finish_trajectory(last_value)
                    sum_return += episode_return
                    sum_length += episode_length
                    num_episodes += 1
                    observation, episode_return, episode_length = self.env.reset(), 0, 0

            # Get values from the buffer
            (
                observation_buffer,
                action_buffer,
                advantage_buffer,
                return_buffer,
                logprobability_buffer,
            ) = self.buffer.get()

            # Update the policy and implement early stopping using KL divergence
            for _ in range(self.train_policy_iterations):
                kl = self.train_policy(
                    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
                )
                if kl > 1.5 * self.target_kl:
                    # Early Stopping
                    break

            # Update the value function
            for _ in range(self.train_value_iterations):
                self.train_value_function(observation_buffer, return_buffer)

            # Print mean return and length for each epoch
            print(
                f" Time:{time.time() - time_start} Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
            )
            if time.time() - time_start > self.max_time:
                break
        return sum_return / num_episodes
