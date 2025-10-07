import numpy as np
from typing import Callable, List, Tuple

class Sarsa0LinearAgent:
    """
    Contrôle on-policy (ε-greedy) avec SARSA(0) et approximation linéaire:
    Q(s,a) ≈ w[a]^T φ(s), w est de taille (nA, d).
    """
    def __init__(self, env, phi: Callable[[int], np.ndarray],
                 gamma: float = 0.99, alpha: float = 0.1,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 episodes: int = 1000, seed: int = 0):
        self.env = env
        self.phi = phi
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.episodes = int(episodes)
        self.rng = np.random.RandomState(seed)

        self.nS = env.n_states
        self.nA = env.action_space
        self.d  = int(len(phi(0)))
        self.w  = np.zeros((self.nA, self.d), dtype=float)  # un vecteur par action

        if getattr(env, "moving_goal", False):
            print("[SARSA lin] Avertissement: moving_goal=True rend l’MDP non-stationnaire.")

    def Q(self, s: int) -> np.ndarray:
        # renvoie le vecteur [Q(s,0),...,Q(s,nA-1)]
        phi_s = self.phi(s)          # (d,)
        return self.w @ phi_s         # (nA,)

    def act_eps_greedy(self, s: int) -> int:
        if self.rng.rand() < self.epsilon:
            return int(self.rng.randint(self.nA))
        q = self.Q(s)
        return int(np.argmax(q))

    def train(self) -> List[float]:
        history = []
        for _ in range(self.episodes):
            s = self.env.reset()
            a = self.act_eps_greedy(s)
            done = False
            ep_ret = 0.0

            while not done:
                s2, r, done = self.env.step(a)
                ep_ret += r
                phi_s = self.phi(s)

                if done:
                    target = r
                else:
                    a2 = self.act_eps_greedy(s2)           # on-policy
                    target = r + self.gamma * self.Q(s2)[a2]

                td_error = target - self.Q(s)[a]
                self.w[a] += self.alpha * td_error * phi_s

                s, a = s2, (a2 if not done else a)

            # mise à jour epsilon & suivi
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            history.append(ep_ret)

        return history

    def greedy_policy(self) -> np.ndarray:
        return np.argmax([self.Q(s) for s in range(self.nS)], axis=1)

    def q_values_table(self) -> np.ndarray:
        Qtab = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            Qtab[s] = self.Q(s)
        return Qtab
