# --- hack simple pour rendre les imports fonctionnels quand on lance le script directement ---
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from envs.gridworld import GridEnv
from agents.agentQL import QLearningAgent
from utils.plotting import ensure_dir, moving_average

# Dossier de sortie
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "sensitivity_grid_size")
ensure_dir(SAVE_DIR)

def eps_schedule(eps0: float, decay: float, eps_min: float, episodes: int):
    """Courbe planifiée d'ε pour visualiser Exploration vs Exploitation (théorique)."""
    eps = eps0 * (decay ** np.arange(episodes))
    return np.maximum(eps, eps_min)

def build_env(n: int) -> GridEnv:
    """
    Construit un environnement n x n avec :
    - start fixe à 0
    - goal fixe en bas à droite
    - ~10% d'obstacles aléatoires (fixes pour une taille donnée)
    Récompenses adaptées : step=-1, goal=+50.
    """
    rows = cols = n
    start = 0
    goals = [rows * cols - 1]
    rng = np.random.RandomState(123)  # reproductible
    k = max(1, int(0.10 * rows * cols))  # ~10%
    all_cells = [i for i in range(rows * cols) if i not in (start, goals[0])]
    obstacles = list(rng.choice(all_cells, size=k, replace=False))

    return GridEnv(
        rows=rows, cols=cols,
        obstacles=obstacles, goals=goals, start=start, seed=123,
        reward_step=-1, reward_goal=50,
        moving_goal=False
    )

def greedy_curve_from_logs(logs: dict, episodes: int) -> np.ndarray:
    """
    Reconstruit la proportion d'actions greedy par épisode à partir des logs StepLogger.
    Attend:
      - logs["step_greedy"] : liste/array de 0/1 par step
      - logs["step_episode"] : épisode auquel appartient le step (entier)
    Retour: array de taille `episodes` avec la fraction greedy par épisode (NaN si épisode absent).
    """
    sg = np.asarray(logs.get("step_greedy", []), dtype=float)
    se = np.asarray(logs.get("step_episode", []), dtype=int)
    curve = np.full(episodes, np.nan)
    if sg.size == 0 or se.size == 0:
        return curve
    for ep in range(episodes):
        mask = (se == ep)
        if np.any(mask):
            curve[ep] = float(sg[mask].mean())
    return curve

def run():
    sizes = list(range(4, 11))  # 4x4 -> 10x10
    EPISODES = 2000
    MAX_STEPS = 250
    W = 50  # fenêtre moyenne mobile

    # ---------------- Courbes de convergence ----------------
    plt.figure(figsize=(8, 5))
    finals = []

    for n in sizes:
        env = build_env(n)
        agent = QLearningAgent(
            env, gamma=0.90, alpha=0.20,
            epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, seed=42
        )
        logs = agent.train(episodes=EPISODES, max_steps=MAX_STEPS)

        ma = moving_average(logs["ep_return"], w=W)
        x = np.arange(len(ma)) + (W - 1)
        plt.plot(x, ma, label=f"{n}x{n}")

        finals.append((n,
                       float(np.mean(logs["ep_return"][-200:])),
                       float(np.std(logs["ep_return"][-200:]))))

    plt.title("Convergence (retour moyen) — Sensibilité à la taille de la grille")
    plt.xlabel("Épisodes"); plt.ylabel(f"Retour (moyenne mobile {W})")
    plt.legend(ncol=3)
    out1 = os.path.join(SAVE_DIR, "sensitivity_grid_convergence.png")
    plt.tight_layout(); plt.savefig(out1, dpi=150); plt.close()

    # ---------------- Barres : performance finale ----------------
    ns = [x[0] for x in finals]
    means = [x[1] for x in finals]
    stds  = [x[2] for x in finals]

    plt.figure(figsize=(8, 5))
    plt.bar([str(n) for n in ns], means, yerr=stds, capsize=4)
    plt.title("Performance finale (200 derniers épisodes)")
    plt.xlabel("Taille de grille"); plt.ylabel("Retour moyen")
    out2 = os.path.join(SAVE_DIR, "sensitivity_grid_final.png")
    plt.tight_layout(); plt.savefig(out2, dpi=150); plt.close()

    # ---------------- Longueur d'épisode lissée ----------------
    plt.figure(figsize=(8, 5))
    for n in sizes:
        env = build_env(n)
        agent = QLearningAgent(
            env, gamma=0.90, alpha=0.20,
            epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, seed=42
        )
        logs = agent.train(episodes=EPISODES, max_steps=MAX_STEPS)
        L = np.asarray(logs["ep_length"], dtype=float)
        if len(L) >= W:
            Ls = np.convolve(L, np.ones(W)/W, mode="valid")
            x = np.arange(len(Ls)) + (W - 1)
            plt.plot(x, Ls, label=f"{n}x{n}")
    plt.title(f"Longueur des épisodes (moyenne mobile {W}) — par taille")
    plt.xlabel("Épisodes"); plt.ylabel("Nombre de pas")
    plt.legend(ncol=3)
    out3 = os.path.join(SAVE_DIR, "sensitivity_grid_episode_length.png")
    plt.tight_layout(); plt.savefig(out3, dpi=150); plt.close()

    # ---------------- Exploration vs Exploitation (ε planifié - théorique) ----------------
    eps = eps_schedule(eps0=1.0, decay=0.995, eps_min=0.01, episodes=EPISODES)
    plt.figure(figsize=(8, 5))
    plt.stackplot(np.arange(EPISODES),
                  [eps, 1.0 - eps],
                  labels=["Exploration (ε)", "Exploitation (1-ε)"],
                  alpha=0.85)
    plt.title("Exploration vs Exploitation (planning ε)")
    plt.xlabel("Épisodes"); plt.ylabel("Proportion")
    plt.legend(loc="upper right")
    out4 = os.path.join(SAVE_DIR, "sensitivity_grid_explore_exploit.png")
    plt.tight_layout(); plt.savefig(out4, dpi=150); plt.close()

    # ---------------- Exploration vs Exploitation (empirique) ----------------
    # 7 sous-figures : proportion d’actions greedy mesurée pendant l’apprentissage, par taille
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, n in enumerate(sizes):
        ax = axes[i]
        env = build_env(n)
        agent = QLearningAgent(
            env, gamma=0.90, alpha=0.20,
            epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, seed=42
        )
        logs = agent.train(episodes=EPISODES, max_steps=MAX_STEPS)

        prop = greedy_curve_from_logs(logs, EPISODES)   # proportion greedy par épisode
        if np.isfinite(prop).any():
            # lissage pour la lisibilité
            mask = np.isfinite(prop)
            prop_ma = prop.copy()
            if mask.sum() >= W:
                prop_ma[mask] = np.convolve(prop[mask], np.ones(W)/W, mode="same")
            ax.plot(np.arange(EPISODES), prop_ma, lw=1.6)
        ax.set_title(f"{n}×{n}")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    # masque le 8e subplot si 7 tailles seulement
    if len(sizes) < len(axes):
        axes[-1].axis("off")

    fig.suptitle("Proportion d’actions greedy (empirique) — par taille de grille")
    for ax in axes[:4]:
        ax.set_xlabel("")
    for ax in axes[4:]:
        ax.set_xlabel("Épisodes")
    for j in (0, 4):
        axes[j].set_ylabel("Proportion greedy")

    out5 = os.path.join(SAVE_DIR, "sensitivity_grid_prop_greedy_subplots.png")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(out5, dpi=150); plt.close()

    print("Figures sauvegardées :")
    for p in (out1, out2, out3, out4, out5):
        print(" -", os.path.abspath(p))

if __name__ == "__main__":
    run()
