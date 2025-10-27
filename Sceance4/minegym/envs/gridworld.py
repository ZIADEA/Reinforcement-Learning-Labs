import numpy as np
from typing import List, Tuple, Optional, Union

class GridEnv:
    """
    GridWorld m x n avec obstacles et un ou plusieurs goals.
    Index état = r * cols + c.

    Barème :
      - step cost:             -3
      - tentative obstacle:   -10 (reste sur place)
      - bump mur:              -5 (reste sur place)
      - goal atteint:         +20 (fin d'épisode)

    Nouveau :
      - moving_goal : si True, les goals changent de place après chaque step
      - moving_mode : "random" (défaut) ou "cyclic"
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
        moving_mode: str = "random",   # "random" ou "cyclic"
    ):
        # RNG interne (pour une vraie reproductibilité)
        self.rng = np.random.RandomState(seed)

        self.rows = rows
        self.cols = cols
        self.n_states = rows * cols
        self.action_space = 4

        # Obstacles
        self.obstacles = list(obstacles) if obstacles else []

        # Start
        if start is None:
            self.start = self._sample_free_cell(forbidden=set(self.obstacles))
        else:
            self.start = start
            # s'il tombe sur un obstacle, on ré-échantillonne proprement
            if self.start in self.obstacles:
                self.start = self._sample_free_cell(forbidden=set(self.obstacles))
        self.state = self.start

        # Goals (liste non vide, sans doublons, hors obstacles et hors start)
        self.goals = self._init_goals(goals)

        # Barèmes
        self.reward_step = reward_step
        self.reward_obstacle_attempt = reward_obstacle_attempt
        self.reward_wall_bump = reward_wall_bump
        self.reward_goal = reward_goal

        # Paramètres de déplacement des goals
        self.moving_goal = moving_goal
        self.moving_mode = moving_mode

    # ------------------------- Utilitaires internes -------------------------

    def _sample_free_cell(self, forbidden: set) -> int:
        """Échantillonne une case libre (pas dans `forbidden`)."""
        cand = self.rng.randint(self.n_states)
        while cand in forbidden:
            cand = self.rng.randint(self.n_states)
        return cand

    def _init_goals(self, goals_spec: Optional[Union[int, List[int]]]) -> List[int]:
        """
        Construit une liste de goals valides :
          - pas dans obstacles
          - pas sur start
          - pas de doublons
          - au moins un goal si goals_spec == None
        """
        forbidden = set(self.obstacles)
        forbidden.add(self.start)
        new_goals: List[int] = []

        if goals_spec is None:
            # un goal aléatoire par défaut
            g = self._sample_free_cell(forbidden)
            new_goals.append(g)
        elif isinstance(goals_spec, int):
            g = goals_spec
            if (g in forbidden) or (g in new_goals):
                g = self._sample_free_cell(forbidden.union(new_goals))
            new_goals.append(g)
        else:
            # liste fournie : on nettoie et on complète si besoin
            for g in goals_spec:
                if (g not in forbidden) and (g not in new_goals):
                    new_goals.append(g)
            if len(new_goals) == 0:
                new_goals.append(self._sample_free_cell(forbidden))

        return new_goals

    # -------------------------- API environnement ---------------------------

    def reset(self) -> int:
        self.state = self.start
        return self.state

    def _in_bounds(self, r: int, c: int) -> bool:
        return (0 <= r < self.rows) and (0 <= c < self.cols)

    def _move_goals(self):
        """
        Met à jour la position des goals selon moving_mode, en évitant :
          - obstacles
          - position de l'agent (self.state)
          - doublons entre goals
        Utilise self.rng (reproductible).
        """
        forbidden = set(self.obstacles)
        forbidden.add(self.state)

        if self.moving_mode == "random":
            new_goals: List[int] = []
            for _ in self.goals:
                cand = self._sample_free_cell(forbidden.union(new_goals))
                new_goals.append(cand)
            self.goals = new_goals

        elif self.moving_mode == "cyclic":
            new_goals: List[int] = []
            for g in self.goals:
                new_g = (g + 1) % self.n_states
                while (new_g in forbidden) or (new_g in new_goals):
                    new_g = (new_g + 1) % self.n_states
                new_goals.append(new_g)
            self.goals = new_goals

    def step(self, action: int) -> Tuple[int, float, bool]:
        s = self.state
        r, c = divmod(s, self.cols)

        pr, pc = r, c
        if action == 0:   pc = c - 1   # gauche
        elif action == 1: pc = c + 1   # droite
        elif action == 2: pr = r - 1   # haut
        elif action == 3: pr = r + 1   # bas

        reward = float(self.reward_step)
        ns = s

        # Mur
        if not self._in_bounds(pr, pc):
            reward += self.reward_wall_bump
        else:
            cand = pr * self.cols + pc
            # Obstacle
            if cand in self.obstacles:
                reward += self.reward_obstacle_attempt
            else:
                ns = cand  # déplacement accepté

        self.state = ns
        done = ns in self.goals
        if done:
            reward += self.reward_goal

        # Déplacement des goals si activé (après avoir vérifié le done)
        if self.moving_goal and not done:
            self._move_goals()

        return ns, reward, done

    def coords(self, s: int) -> Tuple[int, int]:
        return divmod(s, self.cols)
