import numpy as np
from typing import List, Tuple, Optional, Union, Set

class GridEnv:
    """
    GridWorld m x n avec obstacles et un ou plusieurs goals.
    Index état = r * cols + c.

    Barème :
      - step cost:             -3
      - tentative obstacle:   -10 (reste sur place)
      - bump mur:              -5 (reste sur place)
      - goal atteint:         +35 (fin d'épisode)

    STATIC :
        - Goal fixe (même position dans tout l’entraînement)
        - Agent start RANDOM à chaque épisode

    MOVING :
        - Goal initial random à chaque épisode
        - Goal bouge à chaque step
        - Agent start random à chaque épisode
    """

    ACTIONS = {0: "←", 1: "→", 2: "↑", 3: "↓"}

    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        obstacles: Optional[List[int]] = None,
        goals: Optional[Union[int, List[int]]] = None,
        start: Optional[int] = None,
        seed: int = 0,
        reward_step: int = -3,
        reward_obstacle_attempt: int = -10,
        reward_wall_bump: int = -5,
        reward_goal: int = +35,
        moving_goal: bool = False,
        moving_mode: str = "random",  # "random" ou "cyclic"
    ):
        self.rng = np.random.RandomState(seed)

        self.rows = rows
        self.cols = cols
        self.n_states = rows * cols
        self.action_space = 4

        self.obstacles = list(obstacles) if obstacles else []

        # START INITIAL (utilisé seulement pour le tout premier épisode)
        if start is None:
            self.start = self._sample_free_cell(forbidden=set(self.obstacles))
        else:
            self.start = start
            if self.start in self.obstacles:
                self.start = self._sample_free_cell(forbidden=set(self.obstacles))

        self.state = self.start

        # GOALS INITIAUX
        self.goals = self._init_goals(goals)

        # POUR STATIC : on garde une copie du(s) goal(s)
        self.initial_goal = None
        if not moving_goal:
            self.initial_goal = list(self.goals)

        # Barèmes
        self.reward_step = reward_step
        self.reward_obstacle_attempt = reward_obstacle_attempt
        self.reward_wall_bump = reward_wall_bump
        self.reward_goal = reward_goal

        # Moving goal
        self.moving_goal = moving_goal
        self.moving_mode = moving_mode

    # ---------------------------------------------------------------------
    # Utilitaires internes
    # ---------------------------------------------------------------------

    def _sample_free_cell(self, forbidden: Set[int]) -> int:
        cand = self.rng.randint(self.n_states)
        while cand in forbidden:
            cand = self.rng.randint(self.n_states)
        return cand

    def _init_goals(self, goals_spec):
        forbidden = set(self.obstacles)
        forbidden.add(self.start)
        new_goals = []

        if goals_spec is None:
            g = self._sample_free_cell(forbidden)
            new_goals.append(g)

        elif isinstance(goals_spec, int):
            g = goals_spec
            if (g in forbidden) or (g in new_goals):
                g = self._sample_free_cell(forbidden.union(new_goals))
            new_goals.append(g)

        else:
            for g in goals_spec:
                if (g not in forbidden) and (g not in new_goals):
                    new_goals.append(g)
            if len(new_goals) == 0:
                new_goals.append(self._sample_free_cell(forbidden))

        return new_goals

    # ---------------------------------------------------------------------
    # RESET AVEC NOUVELLE LOGIQUE STATIC / MOVING
    # ---------------------------------------------------------------------

    def reset(self) -> int:
        if not self.moving_goal:
            # STATIC : agent random à chaque épisode
            forbid = set(self.obstacles).union(self.initial_goal)
            self.start = self._sample_free_cell(forbid)
            self.state = self.start

            # Goal fixe
            self.goals = list(self.initial_goal)

        else:
            # MOVING : start random + goal initial random à chaque épisode
            forbidden = set(self.obstacles)

            # Agent random
            self.start = self._sample_free_cell(forbidden)
            forbidden.add(self.start)

            # Goal initial random différent du start
            g = self._sample_free_cell(forbidden)
            self.goals = [g]

            self.state = self.start

        return self.state

    # ---------------------------------------------------------------------
    # STEP
    # ---------------------------------------------------------------------

    def _in_bounds(self, r: int, c: int):
        return (0 <= r < self.rows) and (0 <= c < self.cols)

    def _move_goals(self):
        forbidden = set(self.obstacles)
        forbidden.add(self.state)

        if self.moving_mode == "random":
            new_goals = []
            for _ in self.goals:
                cand = self._sample_free_cell(forbidden.union(new_goals))
                new_goals.append(cand)
            self.goals = new_goals

        elif self.moving_mode == "cyclic":
            new_goals = []
            for g in self.goals:
                new_g = (g + 1) % self.n_states
                while (new_g in forbidden) or (new_g in new_goals):
                    new_g = (new_g + 1) % self.n_states
                new_goals.append(new_g)
            self.goals = new_goals

    def step(self, action: int):
        s = self.state
        r, c = divmod(s, self.cols)

        pr, pc = r, c
        if action == 0: pc = c - 1
        elif action == 1: pc = c + 1
        elif action == 2: pr = r - 1
        elif action == 3: pr = r + 1

        reward = float(self.reward_step)
        ns = s

        # Mur ?
        if not self._in_bounds(pr, pc):
            reward += self.reward_wall_bump
        else:
            cand = pr * self.cols + pc
            if cand in self.obstacles:
                reward += self.reward_obstacle_attempt
            else:
                ns = cand

        self.state = ns

        done = ns in self.goals
        if done:
            reward += self.reward_goal

        if self.moving_goal and not done:
            self._move_goals()

        return ns, reward, done

    # ---------------------------------------------------------------------

    def coords(self, s: int):
        return divmod(s, self.cols)
