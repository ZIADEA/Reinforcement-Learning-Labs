import numpy as np
from typing import Callable, List

class TD0LinearValuePrediction:
    """
    TD(0) linéaire pour prédire V^π(s) ≈ w^T φ(s) sous une politique fixe π (on-policy).
    moving_goal doit être False (MDP stationnaire).
    """
    def __init__(self, env, phi: Callable[[int], np.ndarray],
                 policy: Callable[[int], int],
                 alpha: float = 0.1, gamma: float = 0.99,
                 episodes: int = 1000):
        self.env = env
        self.phi = phi              # φ(s) -> R^d
        self.policy = policy        # π(s) -> action
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.episodes = int(episodes)
        self.d = int(len(phi(0)))   # dimension des features
        self.w = np.zeros(self.d)   # V(s) = w^T φ(s)

        if getattr(env, "moving_goal", False):
            print("[TD0] Avertissement: moving_goal=True, la prédiction TD suppose un MDP stationnaire.")

    def V(self, s: int) -> float:
        return float(np.dot(self.w, self.phi(s)))

    def train(self) -> List[float]:
        hist = []
        for _ in range(self.episodes):
            s = self.env.reset()
            done = False
            while not done:
                a = self.policy(s)                         # on-policy
                s2, r, done = self.env.step(a)
                phi_s  = self.phi(s)
                target = r if done else r + self.gamma * self.V(s2)
                delta  = target - self.V(s)
                self.w += self.alpha * delta * phi_s
                s = s2
            # suivi : moyenne des valeurs courantes (optionnel)
            vals = [self.V(ss) for ss in range(self.env.n_states)]
            hist.append(float(np.mean(vals)))
        return hist
