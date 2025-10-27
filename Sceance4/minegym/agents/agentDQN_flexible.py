# agents/agentDQN_flexible.py
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

@dataclass
class FlexibleDQNConfig:
    n_states: int
    n_actions: int
    # Réseau
    hidden: Tuple[int, ...] = (64, 64)   # () => linéaire (pas de hidden)
    lr: float = 1e-3
    gamma: float = 0.98
    gradient_clip: float = 10.0

    # Choix de la loss
    loss: str = "mse"        # "mse" ou "huber"
    huber_beta: float = 1.0  # delta de la Huber (SmoothL1)

    # Epsilon-greedy (linéaire sur les steps d'env.)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 8000

    # Replay / Target (pour DQN)
    use_replay: bool = True
    buffer_capacity: int = 50_000
    batch_size: int = 64
    warmup_size: int = 1_000
    train_every: int = 1

    use_target: bool = True
    target_update_every_steps: int = 5_000

    # Divers
    seed: int = 0
    device: str = "cpu"

def _mlp(n_in: int, n_out: int, hidden: Tuple[int, ...]) -> nn.Module:
    layers = []
    prev = n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers += [nn.Linear(prev, n_out)]
    return nn.Sequential(*layers)

class FlexibleDQNAgent:
    def __init__(self, cfg: FlexibleDQNConfig):
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed)
        self.device = torch.device(cfg.device)

        # Online / Target
        self.online = _mlp(cfg.n_states, cfg.n_actions, cfg.hidden).to(self.device)
        self.target = None
        if cfg.use_target:
            self.target = _mlp(cfg.n_states, cfg.n_actions, cfg.hidden).to(self.device)
            self.target.load_state_dict(self.online.state_dict())
            self.target.eval()

        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)

        # Critère de perte configurable
        if cfg.loss.lower() == "mse":
            self.criterion = nn.MSELoss()
        elif cfg.loss.lower() == "huber":
            # Huber = SmoothL1Loss
            self.criterion = nn.SmoothL1Loss(beta=cfg.huber_beta)
        else:
            raise ValueError(f"Loss inconnue: {cfg.loss}. Utilise 'mse' ou 'huber'.")

        # Epsilon schedule
        self.epsilon = cfg.epsilon_start
        self._step_count = 0

        # One-hot identité pré-calculée (perf + dtype sûr)
        self._eye = torch.eye(cfg.n_states, device=self.device)

        # Replay buffer (ring)
        self._use_replay = cfg.use_replay
        if self._use_replay:
            cap = cfg.buffer_capacity
            self._s  = np.zeros((cap,), dtype=np.int32)
            self._a  = np.zeros((cap,), dtype=np.int64)
            self._r  = np.zeros((cap,), dtype=np.float32)
            self._sn = np.zeros((cap,), dtype=np.int32)
            self._dn = np.zeros((cap,), dtype=np.uint8)
            self._idx = 0
            self._size = 0

    # ---------- utils ----------
    @torch.no_grad()
    def _encode_states(self, s_np: np.ndarray) -> torch.Tensor:
        """
        s_np: shape (B,) d'indices d'états (np.int32/64).
        Retour: one-hot float (B, n_states) sur device, via table identité.
        """
        idx = torch.as_tensor(s_np, device=self.device, dtype=torch.long).view(-1)
        return self._eye.index_select(0, idx)

    @torch.no_grad()
    def q_values(self, s: int) -> np.ndarray:
        x = self._encode_states(np.array([s], dtype=np.int64))
        q = self.online(x).cpu().numpy()[0]
        return q

    def params_l2(self) -> float:
        tot = 0.0
        with torch.no_grad():
            for p in self.online.parameters():
                tot += float((p**2).sum().item())
        return float(np.sqrt(tot))

    # ---------- epsilon ----------
    def _update_epsilon(self):
        if self.epsilon > self.cfg.epsilon_end and self.cfg.epsilon_decay_steps > 0:
            frac = min(1.0, self._step_count / float(self.cfg.epsilon_decay_steps))
            self.epsilon = self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)
            self.epsilon = max(self.cfg.epsilon_end, self.epsilon)

    # ---------- policy ----------
    @torch.no_grad()
    def select_action(self, s: int) -> int:
        if self.rng.rand() < self.epsilon:
            return int(self.rng.randint(self.cfg.n_actions))
        q = self.q_values(s)
        return int(np.argmax(q))

    # ---------- replay ----------
    def remember(self, s: int, a: int, r: float, sn: int, done: bool):
        if not self._use_replay:
            return
        i = self._idx
        self._s[i]  = s
        self._a[i]  = a
        self._r[i]  = r
        self._sn[i] = sn
        self._dn[i] = 1 if done else 0
        self._idx = (self._idx + 1) % self.cfg.buffer_capacity
        self._size = min(self._size + 1, self.cfg.buffer_capacity)

    def _sample_batch(self):
        B = min(self.cfg.batch_size, self._size)
        idxs = self.rng.randint(self._size, size=B)
        s  = self._s[idxs]
        a  = self._a[idxs]
        r  = self._r[idxs]
        sn = self._sn[idxs]
        dn = self._dn[idxs].astype(np.float32)
        return s, a, r, sn, dn

    # ---------- train ----------
    def train_step(self, s: int, a: int, r: float, sn: int, done: bool) -> Optional[float]:
        """
        Unifie les deux modes:
          - NAÏF: pas de replay -> backprop immédiat sur (s,a,r,sn,done)
          - DQN:  with replay -> on stocke, puis on entraîne par mini-batch selon warmup/train_every
        Retourne la loss (float) si une descente a eu lieu, sinon None.
        """
        self._step_count += 1
        self._update_epsilon()

        if not self._use_replay:
            # --- Mode NAÏF : mise à jour sur la transition courante
            self.online.train()
            x  = self._encode_states(np.array([s], dtype=np.int64))
            xn = self._encode_states(np.array([sn], dtype=np.int64))

            q   = self.online(x)                     # (1, A)
            qsa = q[0, a]                            # scalaire

            # Cible TD (scalaire) sans gradient
            with torch.no_grad():
                q_next = self.online(xn)             # pas de target
                max_next = float(q_next.max(dim=1).values.item())
                target_val = r if done else (r + self.cfg.gamma * max_next)
                y = torch.tensor(target_val, device=self.device, dtype=torch.float32)

            loss = self.criterion(qsa, y)
            self.opt.zero_grad()
            loss.backward()
            if self.cfg.gradient_clip is not None:
                nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.gradient_clip)
            self.opt.step()
            return float(loss.item())

        # --- Mode DQN : replay + (option target)
        self.remember(s, a, r, sn, done)

        if self._size < self.cfg.warmup_size:
            return None
        if (self._step_count % self.cfg.train_every) != 0:
            return None

        s_b, a_b, r_b, sn_b, dn_b = self._sample_batch()
        x  = self._encode_states(s_b)     # (B, S)
        xn = self._encode_states(sn_b)    # (B, S)
        a_t  = torch.as_tensor(a_b,  device=self.device, dtype=torch.long).view(-1)
        r_t  = torch.as_tensor(r_b,  device=self.device, dtype=torch.float32).view(-1)
        dn_t = torch.as_tensor(dn_b, device=self.device, dtype=torch.float32).view(-1)

        self.online.train()
        q   = self.online(x)               # (B, A)
        qsa = q.gather(1, a_t.view(-1,1)).squeeze(1)   # (B,)

        with torch.no_grad():
            if self.cfg.use_target and self.target is not None:
                q_next = self.target(xn)
            else:
                q_next = self.online(xn)
            max_next = q_next.max(dim=1).values
            y = r_t + (1.0 - dn_t) * (self.cfg.gamma * max_next)

        loss = self.criterion(qsa, y)
        self.opt.zero_grad()
        loss.backward()
        if self.cfg.gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.gradient_clip)
        self.opt.step()

        # sync target périodique (en steps)
        if self.cfg.use_target and self.target is not None:
            if (self._step_count % self.cfg.target_update_every_steps) == 0:
                self.sync_target()

        return float(loss.item())

    @torch.no_grad()
    def sync_target(self):
        if self.target is not None:
            self.target.load_state_dict(self.online.state_dict())
            self.target.eval()
