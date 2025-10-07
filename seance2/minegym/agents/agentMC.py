# minegym/agents/agentMC.py
import os
import json
import numpy as np
from typing import Tuple, Dict, Any, Optional, List

# (Optionnel) logger si tu l'as déjà dans utils.logger; sinon on ignore proprement
try:
    from utils.logger import StepLogger
except Exception:
    StepLogger = None


class MonteCarloAgent:
    """
    Contrôle Monte-Carlo (first-visit) tabulaire avec politique ε-greedy.

    - Politique: ε-greedy sur Q(s,a)
    - Estimation: moyenne des retours G_t pour les premières visites (first-visit MC)
    - Compatible avec GridEnv (obstacles, multi-goals)
    - Évaluation "greedy" fige le moving_goal pour obtenir une métrique stable
    - API save()/load() pour checkpoints

    Checkpoints:
      ckpt_dir/
        └── mc/
            ├── Q.npy
            └── meta.json
    """

    def __init__(self,
                 env,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 seed: int = 0):
        self.env = env
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)

        self.rng = np.random.RandomState(seed)

        self.nS = env.n_states
        self.nA = env.action_space

        # Q et compteurs pour moyenne (first-visit)
        self.Q = np.zeros((self.nS, self.nA), dtype=float)
        self.returns_sum = np.zeros((self.nS, self.nA), dtype=float)
        self.returns_count = np.zeros((self.nS, self.nA), dtype=np.int64)

    # ---------------- Politique ε-greedy ----------------

    def _epsilon_greedy(self, s: int) -> Tuple[int, bool]:
        greedy_a = int(np.argmax(self.Q[s]))
        if self.rng.rand() < self.epsilon:
            a = int(self.rng.randint(self.nA))
            return a, (a == greedy_a)
        return greedy_a, True

    def greedy_action(self, s: int) -> int:
        return int(np.argmax(self.Q[s]))

    def greedy_value(self) -> np.ndarray:
        return np.max(self.Q, axis=1)

    def greedy_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)

    # ---------------- Évaluation (greedy, ε=0) ----------------

    def evaluate_policy(self, episodes: int = 20, max_steps: int = 500) -> float:
        """Retour moyen sur `episodes` en suivant la politique greedy tirée de Q."""
        total = 0.0
        had_moving = getattr(self.env, "moving_goal", False)
        if had_moving:
            self.env.moving_goal = False

        try:
            for _ in range(episodes):
                s = self.env.reset()
                done = False
                steps = 0
                ret = 0.0
                while not done and steps < max_steps:
                    a = int(np.argmax(self.Q[s]))
                    s, r, done = self.env.step(a)
                    ret += r
                    steps += 1
                total += ret
        finally:
            if had_moving:
                self.env.moving_goal = True

        return total / float(episodes)

    # ---------------- Génération d’un épisode ----------------

    def _generate_episode(self, max_steps: int) -> Tuple[List[int], List[int], List[float]]:
        """
        Joue un épisode sous ε-greedy et renvoie (states, actions, rewards).
        """
        states, actions, rewards = [], [], []
        s = self.env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            a, _ = self._epsilon_greedy(s)
            states.append(s)
            actions.append(a)
            s, r, done = self.env.step(a)
            rewards.append(float(r))
            steps += 1

        return states, actions, rewards

    # ---------------- Entraînement MC (first-visit) ----------------

    def train(self,
              episodes: int = 1000,
              max_steps: int = 500,
              eval_every: Optional[int] = 50,
              eval_episodes: int = 20,
              ckpt_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        First-Visit Monte-Carlo Control:
          - On génère un épisode sous π_ε-greedy
          - On remonte les retours G_t
          - On met à jour Q(s,a) par moyenne sur les premières visites
          - On décroît ε

        Sauvegarde best checkpoint (si ckpt_dir) selon la moyenne greedy évaluée périodiquement.
        """
        logger = StepLogger() if StepLogger is not None else None
        ep_returns: List[float] = []
        ep_lengths: List[int] = []
        best_score = -np.inf

        for ep in range(episodes):
            states, actions, rewards = self._generate_episode(max_steps=max_steps)
            T = len(rewards)

            # Retour cumulatif arrière (G_t)
            G = 0.0
            visited_sa = set()  # pour first-visit
            ep_ret = sum(rewards)
            ep_returns.append(ep_ret)
            ep_lengths.append(T)

            if logger is not None:
                # On logge seulement fin d’épisode + quelques pas si tu veux
                pass

            # Processus backward pour calculer G_t et mettre à jour les premières visites
            for t in reversed(range(T)):
                s_t = states[t]
                a_t = actions[t]
                r_tp1 = rewards[t]
                G = self.gamma * G + r_tp1

                if (s_t, a_t) not in visited_sa:
                    visited_sa.add((s_t, a_t))
                    # Moyenne incrémentale: on cumule et on divise
                    self.returns_sum[s_t, a_t] += G
                    self.returns_count[s_t, a_t] += 1
                    self.Q[s_t, a_t] = self.returns_sum[s_t, a_t] / max(1, self.returns_count[s_t, a_t])

            # décroissance epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # évaluation + checkpoint
            if (ckpt_dir is not None) and (eval_every is not None) and ((ep + 1) % eval_every == 0):
                score = self.evaluate_policy(episodes=eval_episodes, max_steps=max_steps)
                if score > best_score:
                    best_score = score
                    self.save(os.path.join(ckpt_dir, "mc", "best"))

        results: Dict[str, Any] = {
            "Q": self.Q.copy(),
            "V_star": self.greedy_value(),
            "pi_star": self.greedy_policy(),
            "ep_return": np.array(ep_returns, dtype=float),
            "ep_length": np.array(ep_lengths, dtype=int),
            "epsilon_last": self.epsilon,
        }
        return results

    # ---------------- Checkpoints ----------------

    def save(self, path_dir: str) -> None:
        os.makedirs(path_dir, exist_ok=True)
        np.save(os.path.join(path_dir, "Q.npy"), self.Q)
        meta = {
            "algo": "first-visit-mc",
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "n_states": self.nS,
            "action_space": self.nA,
            "env": {
                "rows": getattr(self.env, "rows", None),
                "cols": getattr(self.env, "cols", None),
                "obstacles": list(getattr(self.env, "obstacles", [])),
                "goals": list(getattr(self.env, "goals", [])) if hasattr(self.env, "goals") else None,
                "start": getattr(self.env, "start", None),
                "reward_step": getattr(self.env, "reward_step", None),
                "reward_obstacle_attempt": getattr(self.env, "reward_obstacle_attempt", None),
                "reward_wall_bump": getattr(self.env, "reward_wall_bump", None),
                "reward_goal": getattr(self.env, "reward_goal", None),
                "moving_goal": getattr(self.env, "moving_goal", None),
                "moving_mode": getattr(self.env, "moving_mode", None),
            }
        }
        with open(os.path.join(path_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, path_dir: str) -> None:
        self.Q = np.load(os.path.join(path_dir, "Q.npy"))
        meta_path = os.path.join(path_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.gamma = meta.get("gamma", self.gamma)
            self.epsilon = meta.get("epsilon", self.epsilon)
            # (on pourrait aussi reconfigurer l'env si besoin)
