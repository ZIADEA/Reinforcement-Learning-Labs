import numpy as np
from gym1 import GridEnv

class MonteCarloAgent:
    def __init__(self, env, gamma=0.9, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((env.n_states, env.action_space))
        self.returns = [[[] for _ in range(env.action_space)] for _ in range(env.n_states)]

    def train(self, episodes=500):
        """
        Monte Carlo First-Visit — renvoie la somme des récompenses par épisode.
        """
        rewards_history = []
        for _ in range(episodes):
            episode = []
            s = self.env.reset()
            done = False
            while not done:
                if np.random.rand() < self.epsilon:
                    a = np.random.randint(self.env.action_space)
                else:
                    a = np.argmax(self.Q[s])
                ns, r, done = self.env.step(a)
                episode.append((s, a, r))
                s = ns
            # calculer G et mettre à jour Q
            G = 0
            visited = set()
            for t in reversed(range(len(episode))):
                s, a, r = episode[t]
                G = self.gamma * G + r
                if (s, a) not in visited:
                    self.returns[s][a].append(G)
                    self.Q[s, a] = np.mean(self.returns[s][a])
                    visited.add((s, a))
            rewards_history.append(sum([r for (_,_,r) in episode]))
        return rewards_history

    def act(self, state):
        return np.argmax(self.Q[state])
