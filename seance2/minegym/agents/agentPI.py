# minegym/agents/agentPI.py
import numpy as np
from typing import List, Tuple, Optional

class PolicyIterationAgent:
    """
    Policy Iteration pour GridEnv (stationnaire).
    Hypothèses :
      - transitions déterministes identiques à env.step,
      - moving_goal == False (MDP stationnaire).
    """

    def __init__(self, env, gamma: float = 0.9, theta: float = 1e-3, max_eval_sweeps: int = 10_000):
        self.env = env
        self.gamma = float(gamma)
        self.theta = float(theta)
        self.max_eval_sweeps = int(max_eval_sweeps)

        self.nS = env.n_states
        self.nA = env.action_space

        # V et politique initiales
        self.V = np.zeros(self.nS, dtype=float)
        self.policy = np.random.randint(self.nA, size=self.nS)  # politique aléatoire au départ

        # Sanity check : PI requiert un MDP stationnaire
        if getattr(env, "moving_goal", False):
            print("[PolicyIterationAgent] Avertissement : moving_goal=True détecté. "
                  "La Policy Iteration suppose un MDP stationnaire. "
                  "Désactive moving_goal pour des résultats cohérents.")

    # --------------------- Modèle "virtuel" sans effet de bord ---------------------

    def _in_bounds(self, r: int, c: int) -> bool:
        return (0 <= r < self.env.rows) and (0 <= c < self.env.cols)

    def _simulate_step_from(self, s: int, a: int) -> Tuple[int, float, bool]:
        """
        Reproduit la logique de env.step(action) mais à partir d'un état arbitraire s,
        sans modifier env.state. Retourne (next_state, reward, done).
        """
        rows, cols = self.env.rows, self.env.cols
        r, c = divmod(s, cols)

        pr, pc = r, c
        if a == 0:   pc = c - 1    # gauche
        elif a == 1: pc = c + 1    # droite
        elif a == 2: pr = r - 1    # haut
        elif a == 3: pr = r + 1    # bas

        reward = float(self.env.reward_step)
        ns = s

        if not self._in_bounds(pr, pc):
            reward += self.env.reward_wall_bump
        else:
            cand = pr * cols + pc
            if cand in self.env.obstacles:
                reward += self.env.reward_obstacle_attempt
            else:
                ns = cand

        done = (ns in self.env.goals)
        if done:
            reward += self.env.reward_goal

        return ns, reward, done

    def _expected_return(self, s: int, a: int) -> float:
        """
        Q_pi(s,a) à 1-step lookahead (deterministe) pour l’amélioration de politique.
        """
        ns, r, done = self._simulate_step_from(s, a)
        return r + (0.0 if done else self.gamma * self.V[ns])

    # -------------------------- Algorithme Policy Iteration -------------------------

    def train(self) -> List[float]:
        """
        Effectue Policy Iteration jusqu’à convergence.
        Retourne l'historique de la moyenne de V après chaque sweep d'évaluation.
        """
        history = []

        while True:
            # -------- Policy Evaluation --------
            for _ in range(self.max_eval_sweeps):
                delta = 0.0
                for s in range(self.nS):
                    v_old = self.V[s]
                    a = self.policy[s]
                    ns, r, done = self._simulate_step_from(s, a)
                    self.V[s] = r + (0.0 if done else self.gamma * self.V[ns])
                    delta = max(delta, abs(v_old - self.V[s]))
                history.append(float(self.V.mean()))
                if delta < self.theta:
                    break

            # -------- Policy Improvement --------
            policy_stable = True
            for s in range(self.nS):
                old_a = self.policy[s]
                # choisir l'action qui maximise l'espérance r + gamma V(ns)
                q_vals = [self._expected_return(s, a) for a in range(self.nA)]
                self.policy[s] = int(np.argmax(q_vals))
                if old_a != self.policy[s]:
                    policy_stable = False

            if policy_stable:
                break

        return history

    # ------------------------------ API utilitaires --------------------------------

    def act(self, state: int) -> int:
        """Action selon la politique courante (greedy)."""
        return int(self.policy[state])

    def greedy_value(self) -> np.ndarray:
        """Alias pour V* après convergence."""
        return self.V.copy()

    def greedy_policy(self) -> np.ndarray:
        """Alias pour pi* après convergence."""
        return self.policy.copy()
