from keras import Input, Model
from keras.layers import Dense
import keras
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

eps = np.finfo(np.float32).eps.item()


class ActorCriticModel:
    def __init__(self, input_size, layer_sizes, hidden_layer_activation, optimizer, action_size):
        input_model_obs = Input(input_size)

        model = Dense(layer_sizes[0], activation=hidden_layer_activation)(input_model_obs)

        for layer_size in layer_sizes[1:]:
            model = Dense(layer_size, activation=hidden_layer_activation)(model)
        action = Dense(action_size, activation="softmax")(model)
        critic = Dense(1)(model)

        model = Model(inputs=input_model_obs, outputs=[action, critic])
        self.model = model

    def get_action_prob_and_critic_value(self, state):
        return self.model(state)


class RLAlgorithm(ABC):
    @abstractmethod
    def action(self, env, observation):
        pass

    def train(self, env, epochs):
        pass


class RandomActions(RLAlgorithm):
    def action(self, env, observation):
        return env.action_space.sample()


class ActorCriticRL(RLAlgorithm):
    def __init__(self, model, gamma):
        self.model = model
        self.gamma = gamma

    def action(self, env, state):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)
        action_probs, _ = self.model.model(state)
        action = self.get_action_from_prob_dist(action_probs, env.action_space.n)
        return action

    def prepare_state_to_model_input(self, state):
        state = tf.convert_to_tensor(state)
        return tf.expand_dims(state, 0)

    def get_action_from_prob_dist(self, action_probs, action_space):
        return np.random.choice(action_space, p=np.squeeze(action_probs))

    def function(self, x):
        return tf.math.log(x)

    def get_score(self, reward, current_score):
        return 0.05 * reward + (1 - 0.05) * current_score

    def get_expected_values_for_critic(self, rewards_history):
        expected_values_for_critic = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + self.gamma * discounted_sum
            expected_values_for_critic.insert(0, discounted_sum)
        return expected_values_for_critic

    def normalize(self, array):
        array = np.array(array)
        array = (array - np.mean(array)) / (
                np.std(array) + eps)
        array = array.tolist()
        return array

    def get_actor_and_critic_loss(self, history, huber_loss):
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            diff = ret - value
            actor_losses.append(-log_prob * diff)
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )
        return actor_losses, critic_losses

    def train_model(self, actor_losses, critic_losses, tape, optimizer):
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, self.model.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.model.trainable_variables))

    def train(self, env, epochs):
        huber_loss = keras.losses.Huber()
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        action_probs_history = []
        critic_value_history = []
        max_steps_per_episode = 10000
        rewards_history = []
        score = 0
        running_rewards = []
        episode_count = 0
        while True:
            state = env.reset()
            episode_reward = 0
            with tf.GradientTape() as tape:
                for timestep in range(1, max_steps_per_episode):
                    state = self.prepare_state_to_model_input(state)

                    action_probs, critic_value = self.model.get_action_prob_and_critic_value(state)
                    critic_value_history.append(critic_value[0, 0])

                    action = self.get_action_from_prob_dist(action_probs, env.action_space.n)
                    action_probs_history.append(self.function(action_probs[0, action]))

                    state, reward, done, _ = env.step(action)
                    rewards_history.append(reward)
                    episode_reward += reward

                    if done:
                        break

                score = self.get_score(episode_reward, score)

                expected_values_for_critic = self.get_expected_values_for_critic(rewards_history)

                expected_values_for_critic = self.normalize(expected_values_for_critic)

                history = zip(action_probs_history, critic_value_history, expected_values_for_critic)
                actor_losses, critic_losses = self.get_actor_and_critic_loss(history, huber_loss)

                self.train_model(actor_losses, critic_losses, tape, optimizer)

                action_probs_history.clear()
                critic_value_history.clear()
                rewards_history.clear()
            running_rewards.append(score)
            # Log details
            episode_count += 1
            if episode_count % 10 == 0:
                template = "running reward: {:.2f} at episode {}"
                print(template.format(score, episode_count))

            if score > 195:  # Condition to consider the task solved
                print("Solved at episode {}!".format(episode_count))
                break
            if episode_count > epochs:
                break
        return running_rewards
