import os
import statistics

import numpy as np
import tensorflow as tf


class ReinforceWithBaseline:
    def __init__(self, environment, summary_writer=None, load_model_path=None, save_model_path=None):
        self.environment = environment
        self.observation_shape = self.environment.get_observation_shape()
        self.num_actions = self.environment.get_num_actions()
        self.alpha_theta = 0.001
        self.alpha_w = 0.001
        self.discount_factor = 1
        self.eps = np.finfo(np.float32).eps.item()
        self.load_model_path = load_model_path
        self.save_model_path = save_model_path
        self.summary_writer = summary_writer

        self.policy = self.get_nn_policy()
        self.state_value_function = self.get_state_value_function()

    def get_nn_policy(self):
        if self.load_model_path is not None:
            return self.get_saved_nn_policy()

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=256, input_shape=self.observation_shape, activation='relu'))
        # model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(units=256, activation='relu'))
        # model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(units=self.num_actions, activation='softmax'))

        return model

    def get_state_value_function(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=256, input_shape=self.observation_shape, activation='relu'))
        # model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(units=256, activation='relu'))
        # model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(units=1, activation='linear'))

        return model

    def get_saved_nn_policy(self):
        model = tf.keras.models.load_model(self.load_model_path)
        return model

    def get_state_value_function_gradients(self, state):
        with tf.GradientTape() as tape:
            state_value = self.state_value_function(state)[0]

        gradients = tape.gradient(state_value, self.state_value_function.trainable_variables)

        return gradients

    def get_log_policy_gradients(self, state, action):
        with tf.GradientTape() as tape:
            selected_action_log_prob = tf.math.log(self.policy(state)[0][action])

        gradients = tape.gradient(selected_action_log_prob, self.policy.trainable_variables)

        return gradients

    def update_policy_weights(self, gradients, delta, num_step):
        for i in range(len(self.policy.trainable_variables)):
            self.policy.trainable_variables[i].assign_add(
                self.alpha_theta * delta * (self.discount_factor ** num_step) * gradients[i]
            )

    def update_state_value_function_weights(self, gradients, delta):
        for i in range(len(self.state_value_function.trainable_variables)):
            self.state_value_function.trainable_variables[i].assign_add(
                self.alpha_w * delta * gradients[i]
            )

    def get_action(self, observation):
        action_probs = self.policy(observation)[0].numpy()
        action = np.random.choice(a=self.num_actions, p=action_probs)

        return action

    def normalize_returns(self, returns):
        mean = statistics.mean(returns)
        stdev = statistics.stdev(returns)

        normalized_returns = [(x - mean) / (stdev + self.eps) for x in returns]

        return normalized_returns

    def learn_optimal_policy(self, num_epochs=10000):
        for epoch_num in range(num_epochs):
            states = []
            actions = []
            rewards = []
            policy_gradients_list = []
            state_value_gradients_list = []

            done = False
            observation = self.environment.reset()

            while not done:
                observation = tf.expand_dims(observation, 0)
                states.append(observation)

                action = self.get_action(observation)
                actions.append(action)

                observation, reward, done, info = self.environment.step(action)
                rewards.append(reward)

                policy_gradients = self.get_log_policy_gradients(state=states[-1], action=actions[-1])
                policy_gradients_list.append(policy_gradients)

                state_value_gradients = self.get_state_value_function_gradients(state=states[-1])
                state_value_gradients_list.append(state_value_gradients)

            if self.summary_writer:
                self.summary_writer.write_summary("Episode Return", sum(rewards), epoch_num)

            returns = rewards.copy()
            for i in reversed(range(len(rewards) - 1)):
                returns[i] += self.discount_factor * returns[i + 1]

            # normalized_returns = self.normalize_returns(returns)

            deltas = [None] * len(returns)
            for i in range(len(rewards)):
                deltas[i] = returns[i] - self.state_value_function(states[i])[0][0].numpy()

            deltas = self.normalize_returns(deltas)

            for i in range(len(states)):
                self.update_state_value_function_weights(gradients=state_value_gradients_list[i], delta=deltas[i])
                self.update_policy_weights(policy_gradients_list[i], deltas[i], i)

            if (self.save_model_path is not None) and not ((epoch_num + 1) % 1000):
                saved_filename = "_epoch_" + str(epoch_num + 1) + ".h5"
                saved_file_path = os.path.join(self.save_model_path, saved_filename)

                self.policy.save(saved_file_path)
