# minegym/agents/agentVI.py
import numpy as np
from typing import List, Tuple

class ValueIterationAgent:
    """
    Value Iteration pour GridEnv (stationnaire).
    Hypothèses :
      - transitions déterministes identiques à env.step,
      - moving_goal == False (MDP stationnaire).
    """

    def __init__(self, env, gamma: float = 0.9, theta: float = 1e-3, max_iters: int = 1_000_000):
        self.env = env
        self.gamma = float(gamma)
        self.theta = float(theta)
        self.max_iters = int(max_iters)

        self.nS = env.n_states
        self.nA = env.action_space

        self.V = np.zeros(self.nS, dtype=float)
        self.policy = np.zeros(self.nS, dtype=int)

        # Avertir si l'env est non stationnaire
        if getattr(env, "moving_goal", False):
            print("[ValueIterationAgent] Avertissement : moving_goal=True détecté. "
                  "La Value Iteration suppose un MDP stationnaire. "
                  "Désactive moving_goal pour des résultats cohérents.")

    # --------------------- Modèle "virtuel" sans effet de bord ---------------------

    def _in_bounds(self, r: int, c: int) -> bool:
        return (0 <= r < self.env.rows) and (0 <= c < self.env.cols)

    def _simulate_step_from(self, s: int, a: int) -> Tuple[int, float, bool]:
        """
        Reproduit la logique de env.step(action) à partir d’un état arbitraire s,
        sans changer env.state. Retourne (next_state, reward, done).
        """
        rows, cols = self.env.rows, self.env.cols
        r, c = divmod(s, cols)

        pr, pc = r, c
        if a == 0:   pc = c - 1   # gauche
        elif a == 1: pc = c + 1   # droite
        elif a == 2: pr = r - 1   # haut
        elif a == 3: pr = r + 1   # bas

        reward = float(self.env.reward_step)
        ns = s

        # mur
        if not self._in_bounds(pr, pc):
            reward += self.env.reward_wall_bump
        else:
            cand = pr * cols + pc
            # obstacle
            if cand in self.env.obstacles:
                reward += self.env.reward_obstacle_attempt
            else:
                ns = cand

        done = (ns in self.env.goals)
        if done:
            reward += self.env.reward_goal

        return ns, reward, done

    def _q_value(self, s: int, a: int) -> float:
        ns, r, done = self._simulate_step_from(s, a)
        return r + (0.0 if done else self.gamma * self.V[ns])

    # ---------------------------- Value Iteration ----------------------------

    def train(self) -> list:
        """
        Exécute Value Iteration jusqu’à convergence.
        Retourne l'historique de la moyenne de V après chaque itération (pour tracer).
        """
        history = []
        for _ in range(self.max_iters):
            delta = 0.0
            # mise à jour synchronisée (copie de V)
            V_new = self.V.copy()

            for s in range(self.nS):
                # V(s) = max_a Q(s,a)
                q_vals = [self._q_value(s, a) for a in range(self.nA)]
                best = max(q_vals)
                delta = max(delta, abs(self.V[s] - best))
                V_new[s] = best

            self.V = V_new
            history.append(float(self.V.mean()))
            if delta < self.theta:
                break

        # extraire la politique optimale π*(s) = argmax_a Q(s,a)
        for s in range(self.nS):
            q_vals = [self._q_value(s, a) for a in range(self.nA)]
            self.policy[s] = int(np.argmax(q_vals))

        return history

    # ------------------------------ API utilitaires --------------------------------

    def act(self, state: int) -> int:
        """Action selon la politique optimale (greedy)."""
        return int(self.policy[state])

    def greedy_value(self) -> np.ndarray:
        """Renvoie V* après convergence."""
        return self.V.copy()

    def greedy_policy(self) -> np.ndarray:
        """Renvoie π* après convergence."""
        return self.policy.copy()
