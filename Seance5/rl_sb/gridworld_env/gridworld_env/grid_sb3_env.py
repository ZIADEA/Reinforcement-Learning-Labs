import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple

from .grid_core import GridEnv

class SB3GridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        obstacles=None,
        goals=None,
        start: Optional[int] = None,
        seed: int = 0,
        moving_goal: bool = False,
        moving_mode: str = "random",
    ):
        super().__init__()

        self.core = GridEnv(
            rows=rows,
            cols=cols,
            obstacles=obstacles,
            goals=goals,
            start=start,
            seed=seed,
            moving_goal=moving_goal,
            moving_mode=moving_mode,
        )

        self.observation_space = spaces.Discrete(self.core.n_states)
        self.action_space = spaces.Discrete(self.core.action_space)

        self._max_steps = 100
        self._step_count = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.core.rng = np.random.RandomState(seed)

        self._step_count = 0
        s = self.core.reset()
        return int(s), {}

    def step(self, action: int):
        self._step_count += 1

        ns, reward, done = self.core.step(int(action))

        terminated = bool(done)
        truncated = self._step_count >= self._max_steps

        info = {"goals": list(self.core.goals)}

        return int(ns), float(reward), terminated, truncated, info

    def render(self):
        self.core.render()
