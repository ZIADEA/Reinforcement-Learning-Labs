import numpy as np
from gym1 import GridEnv

class QLearningAgent:
    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = np.zeros((env.n_states, env.action_space))

    def train(self, episodes=500):
        """
        Q-Learning — renvoie la somme des récompenses par épisode.
        """
        rewards_history = []
        for _ in range(episodes):
            s = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                if np.random.rand() < self.epsilon:
                    a = np.random.randint(self.env.action_space)
                else:
                    a = np.argmax(self.Q[s])
                ns, r, done = self.env.step(a)
                self.Q[s, a] += self.alpha * (r + self.gamma * np.max(self.Q[ns]) - self.Q[s, a])
                s = ns
                total_reward += r
            rewards_history.append(total_reward)
        return rewards_history

    def act(self, state):
        return np.argmax(self.Q[state])
