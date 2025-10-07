import os
import json
import numpy as np
from typing import Tuple, Dict, Any, Optional
from utils.logger import StepLogger


class QLearningAgent:
    """
    Agent Q-Learning tabulaire avec:
      - logging pas-à-pas (StepLogger)
      - évaluation greedy périodique
      - sauvegarde automatique du meilleur modèle (best checkpoint)
      - API save()/load() pour Q et hyperparamètres

    Checkpoints:
      ckpt_dir/
        └── best/
            ├── Q.npy
            └── meta.json
    """

    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, seed: int = 0):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = np.random.RandomState(seed)
        self.Q = np.zeros((env.n_states, env.action_space), dtype=float)

    # -------------------------
    #  API: sauvegarde / charge
    # -------------------------
    def save(self, path_dir: str) -> None:
        os.makedirs(path_dir, exist_ok=True)
        np.save(os.path.join(path_dir, "Q.npy"), self.Q)
        meta = {
            "gamma": self.gamma,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "n_states": self.env.n_states,
            "action_space": self.env.action_space,
            # contexte utile pour rejouer
            "env": {
                "rows": getattr(self.env, "rows", None),
                "cols": getattr(self.env, "cols", None),
                "obstacles": list(getattr(self.env, "obstacles", [])),
                "goals": list(getattr(self.env, "goals", [])) if hasattr(self.env, "goals") else [getattr(self.env, "goal", None)],
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
            # on synchronise quelques hyperparamètres (facultatif)
            self.gamma = meta.get("gamma", self.gamma)
            self.alpha = meta.get("alpha", self.alpha)
            self.epsilon = meta.get("epsilon", self.epsilon)

    # -------------
    #  Politique π*
    # -------------
    def _epsilon_greedy(self, s: int) -> Tuple[int, bool]:
        greedy_a = int(np.argmax(self.Q[s]))
        if self.rng.rand() < self.epsilon:
            a = int(self.rng.randint(self.env.action_space))
            # aléatoire peut tomber sur l'action gourmande
            return a, (a == greedy_a)
        return greedy_a, True  # action gourmande

    def greedy_action(self, s: int) -> int:
        return int(np.argmax(self.Q[s]))

    def greedy_value(self):
        return np.max(self.Q, axis=1)

    def greedy_policy(self):
        return np.argmax(self.Q, axis=1)

    # -----------------------
    #  Évaluation (greedy ε=0)
    # -----------------------
    def evaluate_policy(self, episodes: int = 20, max_steps: int = 300) -> float:
        """Retourne le retour moyen sur `episodes` (policy greedy)."""
        total = 0.0
        for _ in range(episodes):
            s = self.env.reset()
            done = False
            steps = 0
            ep_ret = 0.0
            while not done and steps < max_steps:
                a = int(np.argmax(self.Q[s]))  # greedy
                s, r, done = self.env.step(a)
                ep_ret += r
                steps += 1
            total += ep_ret
        return total / float(episodes)

    # ----------------
    #  Entraînement
    # ----------------
    def train(self,
              episodes: int = 500,
              max_steps: int = 300,
              # évaluation/sauvegarde best:
              eval_every: Optional[int] = 50,
              eval_episodes: int = 20,
              ckpt_dir: Optional[str] = None,
              # alternative sans éval: déclencher sur moyenne mobile d'entraînement
              trigger_on_moving_avg: bool = False,
              ma_window: int = 50
              ) -> Dict[str, Any]:
        """
        Si `eval_every` est défini et `ckpt_dir` non nul:
          - Tous les `eval_every` épisodes, évalue la politique greedy,
            et sauvegarde le best checkpoint si score en hausse.

        Sinon, si `trigger_on_moving_avg=True`:
          - Sauvegarde quand la moyenne mobile des retours d'entraînement (fenêtre `ma_window`)
            s'améliore.

        Retourne un dict de logs + Q/V*/π* finaux.
        """
        logger = StepLogger()
        rewards_history = []
        best_score = -np.inf

        for ep in range(episodes):
            s = self.env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done and steps < max_steps:
                a, is_greedy = self._epsilon_greedy(s)
                ns, r, done = self.env.step(a)

                td_target = r + (0 if done else self.gamma * np.max(self.Q[ns]))
                td_error = td_target - self.Q[s, a]
                self.Q[s, a] += self.alpha * td_error

                logger.log(
                    step_global=logger.total_steps,
                    episode=ep,
                    step_in_ep=steps,
                    state=s,
                    action=a,
                    reward=r,
                    epsilon=self.epsilon,
                    greedy=is_greedy,
                    done=done,
                )

                s = ns
                total_reward += r
                steps += 1

            # fin épisode
            rewards_history.append(total_reward)
            logger.close_episode(total_reward, steps)

            # décroissance epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # --- Déclencheurs de sauvegarde best ---
            did_save = False

            # A) Critère basé sur évaluation séparée (recommandé)
            if (ckpt_dir is not None) and (eval_every is not None) and ((ep + 1) % eval_every == 0):
                score = self.evaluate_policy(episodes=eval_episodes, max_steps=max_steps)
                if score > best_score:
                    best_score = score
                    self.save(os.path.join(ckpt_dir, "best"))
                    did_save = True

            # B) Critère alternatif basé sur moyenne mobile d'entraînement
            if (not did_save) and trigger_on_moving_avg and (ckpt_dir is not None):
                if len(rewards_history) >= ma_window:
                    curr_ma = float(np.mean(rewards_history[-ma_window:]))
                    if curr_ma > best_score:
                        best_score = curr_ma
                        self.save(os.path.join(ckpt_dir, "best_ma"))

        results = logger.to_dict()
        results["Q"] = self.Q.copy()
        results["V_star"] = np.max(self.Q, axis=1)
        results["pi_star"] = np.argmax(self.Q, axis=1)
        results["epsilon_last"] = self.epsilon
        return results
