import math
import random
from typing import Tuple, Deque, Callable, Optional, Dict, Any
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -------------------------
# Encodages d'état (features)
# -------------------------

def make_phi_one_hot(n_states: int) -> Callable[[int], np.ndarray]:
    """φ(s) = one-hot de taille n_states."""
    def phi(s: int) -> np.ndarray:
        v = np.zeros(n_states, dtype=np.float32)
        v[s] = 1.0
        return v
    return phi

def make_phi_coords_bias(rows: int, cols: int) -> Callable[[int], np.ndarray]:
    """φ(s) = [row_norm, col_norm, bias]. Très compact (3 dims)."""
    def phi(s: int) -> np.ndarray:
        r, c = divmod(s, cols)
        R = 0.0 if rows <= 1 else r / (rows - 1)
        C = 0.0 if cols <= 1 else c / (cols - 1)
        return np.array([R, C, 1.0], dtype=np.float32)
    return phi


# -------------------------
# Modèle Q (MLP 1 couche cachée)
# -------------------------

class QNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------
# Replay buffer très simple
# -------------------------

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.stack(s).astype(np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2).astype(np.float32),
            np.array(d, dtype=np.bool_)
        )

    def __len__(self):
        return len(self.buf)


# -------------------------
# Agent DQN (1 hidden layer)
# -------------------------

class DQNAgent:
    """
    DQN minimal :
      - MLP une seule couche cachée
      - ε-greedy
      - Replay Buffer
      - Réseau cible (target network)
      - Perte MSE sur cible TD:  y = r + γ * (1-done) * max_a' Q_target(s', a')
    Conseillé: env.moving_goal == False pour rester stationnaire.
    """
    def __init__(
        self,
        env,
        phi_fn: Optional[Callable[[int], np.ndarray]] = None,
        hidden: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_episodes: int = 1000,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        target_update_every: int = 200,   # en pas (steps)
        seed: int = 0,
        device: Optional[str] = None,
    ):
        self.env = env
        self.nA = env.action_space
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_every = int(target_update_every)

        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        # Encodage d'état
        if phi_fn is None:
            # par défaut : coordonnées + biais (3 dims). Pour de petits MDP, one-hot marche très bien.
            phi_fn = make_phi_coords_bias(env.rows, env.cols)
        self.phi = phi_fn

        # Dimensions
        in_dim = len(self.phi(0))

        # Réseaux
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.qnet = QNet(in_dim, self.nA, hidden).to(self.device)
        self.target = QNet(in_dim, self.nA, hidden).to(self.device)
        self.target.load_state_dict(self.qnet.state_dict())
        self.target.eval()

        self.optim = optim.Adam(self.qnet.parameters(), lr=lr)
        self.criteria = nn.MSELoss()

        # Replay
        self.replay = ReplayBuffer(buffer_capacity)

        # ε planning (linéaire sur les épisodes)
        self.eps_start = float(epsilon_start)
        self.eps_end   = float(epsilon_end)
        self.eps_decay_episodes = int(epsilon_decay_episodes)

        self._global_step = 0

        if getattr(env, "moving_goal", False):
            print("[DQN] Avertissement : moving_goal=True => non-stationnaire. DQN peut rester stable grâce au réseau cible,"
                  " mais la convergence n’est pas garantie.")

    # -------- utils --------

    def _epsilon_for_episode(self, ep: int) -> float:
        # décroissance linéaire d'ε sur les 'epsilon_decay_episodes' premiers épisodes
        t = min(ep / max(1, self.eps_decay_episodes), 1.0)
        return (1.0 - t) * self.eps_start + t * self.eps_end

    def _phi_t(self, s: int) -> torch.Tensor:
        arr = self.phi(s)
        return torch.from_numpy(arr).to(self.device)

    def _act_eps_greedy(self, s: int, eps: float) -> int:
        if self.rng.rand() < eps:
            return int(self.rng.randint(self.nA))
        with torch.no_grad():
            q = self.qnet(self._phi_t(s))
            return int(torch.argmax(q).item())

    # -------- entraînement --------

    def train(self,
              episodes: int = 1000,
              max_steps: int = 250,
              warmup_steps: int = 1000) -> Dict[str, Any]:

        ep_return_list, ep_length_list = [], []
        greedy_step_flags = []  # pour mesurer proportion greedy
        eps_used = []

        for ep in range(episodes):
            s = self.env.reset()
            eps = self._epsilon_for_episode(ep)
            eps_used.append(eps)

            done = False
            total = 0.0
            t = 0

            while not done and t < max_steps:
                # action
                a_greedy = self._act_eps_greedy(s, 0.0)
                a = self._act_eps_greedy(s, eps)
                greedy_step_flags.append(1 if a == a_greedy else 0)

                # transition réelle de l'env
                s2, r, done = self.env.step(a)

                # stocker dans le replay (états encodés)
                self.replay.push(self.phi(s), a, r, self.phi(s2), done)

                # update DQN si assez d’exemples
                if len(self.replay) >= self.batch_size and self._global_step > warmup_steps:
                    self._learn_step()

                # update réseau cible périodiquement
                if (self._global_step % self.target_update_every) == 0:
                    self.target.load_state_dict(self.qnet.state_dict())

                s = s2
                total += r
                t += 1
                self._global_step += 1

            ep_return_list.append(total)
            ep_length_list.append(t)

        return {
            "ep_return": np.asarray(ep_return_list, dtype=np.float32),
            "ep_length": np.asarray(ep_length_list, dtype=np.int32),
            "step_greedy": np.asarray(greedy_step_flags, dtype=np.int8),
            "eps_used": np.asarray(eps_used, dtype=np.float32),
        }

    def _learn_step(self):
        S, A, R, S2, D = self.replay.sample(self.batch_size)

        S  = torch.from_numpy(S).to(self.device)   # (B, d)
        A  = torch.from_numpy(A).to(self.device)   # (B,)
        R  = torch.from_numpy(R).to(self.device)   # (B,)
        S2 = torch.from_numpy(S2).to(self.device)  # (B, d)
        D  = torch.from_numpy(D.astype(np.float32)).to(self.device)  # (B,)

        # Q(s,a)
        q_all = self.qnet(S)                       # (B, nA)
        q_sa  = q_all.gather(1, A.view(-1, 1)).squeeze(1)  # (B,)

        # y = r + γ (1-done) max_a' Q_target(s', a')
        with torch.no_grad():
            q_next_all = self.target(S2)           # (B, nA)
            q_next_max = q_next_all.max(dim=1).values
            y = R + self.gamma * (1.0 - D) * q_next_max

        loss = self.criteria(q_sa, y)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=10.0)
        self.optim.step()

    # -------- évaluation --------

    def evaluate_greedy(self, episodes: int = 10, max_steps: int = 250) -> float:
        total = 0.0
        self.qnet.eval()
        with torch.no_grad():
            for _ in range(episodes):
                s = self.env.reset()
                done = False
                steps = 0
                G = 0.0
                while not done and steps < max_steps:
                    a = int(torch.argmax(self.qnet(self._phi_t(s))).item())
                    s, r, done = self.env.step(a)
                    G += r
                    steps += 1
        self.qnet.train()
        return total / max(1, episodes)
