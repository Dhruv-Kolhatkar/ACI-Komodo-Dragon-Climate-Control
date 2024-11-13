import numpy as np
import random

class KomodoEnvironmentModel:
    def __init__(self):
        self.state_space = 1000  # Discretized state space
        self.action_space = 27   # 3 actions for each parameter (increase, decrease, no change)
        self.q_table = np.zeros((self.state_space, self.action_space))
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1

    def get_state(self, temperature, humidity, wind):
        # Discretize the continuous state space
        t = int(temperature * 10)
        h = int(humidity * 10)
        w = int(wind * 10)
        return (t * 100 + h * 10 + w) % self.state_space

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_next_q)
        self.q_table[state, action] = new_q

    def get_next_weather(self, temperature, humidity, wind):
        state = self.get_state(temperature, humidity, wind)
        action = self.choose_action(state)
        
        # Interpret the action
        t_change, h_change, w_change = action // 9, (action % 9) // 3, action % 3
        t_change = (t_change - 1) * 0.1
        h_change = (h_change - 1) * 0.1
        w_change = (w_change - 1) * 0.1

        new_temp = max(0, min(40, temperature + t_change))
        new_humidity = max(0, min(100, humidity + h_change))
        new_wind = max(0, min(100, wind + w_change))

        return new_temp, new_humidity, new_wind

    def train(self, state, action, reward, next_state):
        self.update_q_table(state, action, reward, next_state)