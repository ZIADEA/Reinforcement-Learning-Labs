import numpy as np
from gym1 import GridEnv

class PolicyIterationAgent:
    def __init__(self, env, gamma=0.9, theta=1e-3):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros(env.n_states)
        self.policy = np.random.randint(env.action_space, size=env.n_states)

    def train(self):
        """
        Policy Iteration — renvoie une liste de la moyenne de V après chaque sweep.
        """
        history = []
        while True:
            # -------- Policy Evaluation --------
            while True:
                delta = 0
                for s in range(self.env.n_states):
                    v = self.V[s]
                    a = self.policy[s]
                    ns, r, _ = self.simulate_step(s, a)
                    self.V[s] = r + self.gamma * self.V[ns]
                    delta = max(delta, abs(v - self.V[s]))
                history.append(np.mean(self.V))
                if delta < self.theta:
                    break

            # -------- Policy Improvement --------
            policy_stable = True
            for s in range(self.env.n_states):
                old_a = self.policy[s]
                self.policy[s] = np.argmax([self.expected_return(s, a) for a in range(self.env.action_space)])
                if old_a != self.policy[s]:
                    policy_stable = False
            if policy_stable:
                break
        return history

    def simulate_step(self, s, a):
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
