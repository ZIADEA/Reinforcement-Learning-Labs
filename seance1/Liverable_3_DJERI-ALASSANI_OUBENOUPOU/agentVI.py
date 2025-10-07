import numpy as np
from gym1 import GridEnv

class ValueIterationAgent:
    def __init__(self, env, gamma=0.9, theta=1e-3):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros(env.n_states)
        self.policy = np.zeros(env.n_states, dtype=int)

    def train(self):
        """
        Exécute Value Iteration et retourne une liste avec la moyenne de V après chaque itération
        pour pouvoir tracer la convergence.
        """
        history = []
        while True:
            delta = 0
            for s in range(self.env.n_states):
                v = self.V[s]
                values = []
                for a in range(self.env.action_space):
                    ns, r, _ = self.simulate_step(s, a)
                    values.append(r + self.gamma * self.V[ns])
                self.V[s] = max(values)
                delta = max(delta, abs(v - self.V[s]))
            history.append(np.mean(self.V))
            if delta < self.theta:
                break
        # extraire policy
        for s in range(self.env.n_states):
            self.policy[s] = np.argmax([self.expected_return(s, a) for a in range(self.env.action_space)])
        return history

    def simulate_step(self, s, a):
        # copie logique de step mais sans changer self.state
        size = self.env.size
        ns = s
        if a == 0 and s % size != 0:            ns -= 1
        elif a == 1 and s % size != size - 1:   ns += 1
        elif a == 2 and s - size >= 0:          ns -= size
        elif a == 3 and s + size < size*size:   ns += size
        r = 10 if ns == self.env.goal else 0
        return ns, r, ns == self.env.goal

    def expected_return(self, s, a):
        ns, r, _ = self.simulate_step(s, a)
        return r + self.gamma * self.V[ns]

    def act(self, state):
        return self.policy[state]
