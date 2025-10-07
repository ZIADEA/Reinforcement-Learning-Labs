# --- hack simple pour rendre les imports fonctionnels quand on lance le script directement ---
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

from envs.gridworld import GridEnv
from agents.agentQL import QLearningAgent
from utils.plotting import ensure_dir

# Paramètres globaux
SEEDS = [0, 1, 2, 3, 4]
GAMMAS = [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)] + [0.0, 0.99]  # 11 gammas
EPISODES = 2000
MAX_STEPS = 250
WINDOW = 50
THRESHOLD = -10.0

# Dossier de sortie
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "sensitivity_gamma")
ensure_dir(SAVE_DIR)


def eps_schedule(eps0: float, decay: float, eps_min: float, episodes: int):
    eps = eps0 * (decay ** np.arange(episodes))
    return np.maximum(eps, eps_min)


def fixed_env() -> GridEnv:
    """
    Environnement 10x10, obstacles en diagonale, goal en bas à droite.
    Récompenses adaptées : step=-1, goal=+50.
    """
    rows, cols = 10, 10
    start = 0
    goals = [rows * cols - 1]
    obstacles = []
    for i in range(1, min(rows, cols) - 1):
        obstacles.append(i * cols + (i - 1))
    return GridEnv(
        rows=rows, cols=cols,
        obstacles=obstacles, goals=goals, start=start, seed=123,
        reward_step=-1, reward_goal=50,
        moving_goal=False
    )


def compute_prop_greedy_per_episode(logs: Dict[str, np.ndarray]) -> np.ndarray:
    """Retourne la proportion d’actions greedy par épisode à partir des logs step-by-step."""
    greedy = np.asarray(logs["step_greedy"], dtype=float)       # 0/1 par step
    ep_idx = np.asarray(logs["step_episode"], dtype=int)        # index d’épisode par step
    E = int(len(logs["ep_length"]))                             # nb d’épisodes
    prop = np.zeros(E, dtype=float)
    for e in range(E):
        mask = (ep_idx == e)
        prop[e] = greedy[mask].mean() if mask.any() else np.nan
    return prop


def run_one(gamma: float, seed: int) -> Dict[str, np.ndarray]:
    """
    Exécute un run complet pour (gamma, seed) et renvoie:
      - ep_return : (E,)
      - ep_length : (E,)
      - prop_greedy : (E,) proportion d’actions greedy par épisode
    """
    env = fixed_env()
    agent = QLearningAgent(
        env, gamma=gamma, alpha=0.20,
        epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, seed=seed
    )
    logs = agent.train(episodes=EPISODES, max_steps=MAX_STEPS)
    ep_return = np.asarray(logs["ep_return"], dtype=float)
    ep_length = np.asarray(logs["ep_length"], dtype=float)
    prop_greedy = compute_prop_greedy_per_episode(logs)
    return {"ep_return": ep_return, "ep_length": ep_length, "prop_greedy": prop_greedy}


def mean_ci(arrs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moyenne et IC95% (approx gaussienne) pour une liste de courbes."""
    L = min(len(a) for a in arrs)
    M = np.stack([a[:L] for a in arrs], axis=0)
    mean = M.mean(axis=0)
    std  = M.std(axis=0)
    ci = 1.96 * std / np.sqrt(M.shape[0])
    return mean, mean - ci, mean + ci


def time_to_threshold(curve: np.ndarray, thr: float, w: int = 50) -> int:
    """Premier épisode où la MA(w) dépasse `thr` (ou len(curve) si jamais atteint)."""
    if len(curve) < w:
        return len(curve)
    ma = np.convolve(curve, np.ones(w)/w, mode="valid")
    if np.any(ma >= thr):
        idx = int(np.argmax(ma >= thr))
        return idx + w - 1
    return len(curve)


def run():
    # -------- Apprentissage multi-seeds pour chaque gamma --------
    results = {}
    for g in GAMMAS:
        ep_returns, ep_lengths, prop_greedies, ttts = [], [], [], []
        for sd in SEEDS:
            out = run_one(g, sd)
            ep_returns.append(out["ep_return"])
            ep_lengths.append(out["ep_length"])
            prop_greedies.append(out["prop_greedy"])
            ttts.append(time_to_threshold(out["ep_return"], THRESHOLD, w=WINDOW))
        results[g] = {
            "ep_return": ep_returns,
            "ep_length": ep_lengths,
            "prop_greedy": prop_greedies,
            "ttt": np.asarray(ttts, dtype=float),
        }

    # -------- Fig 1 : Convergence moyenne (MA WINDOW) --------
    plt.figure(figsize=(9, 5))
    for g in GAMMAS:
        mean, lo, hi = mean_ci(results[g]["ep_return"])
        ma = np.convolve(mean, np.ones(WINDOW)/WINDOW, mode="valid")
        x = np.arange(len(ma)) + (WINDOW - 1)
        plt.plot(x, ma, label=f"γ={g:.2f}", linewidth=1.8)
        # Option bandes d’incertitude :
        # lo_ma = np.convolve(lo, np.ones(WINDOW)/WINDOW, mode="valid")
        # hi_ma = np.convolve(hi, np.ones(WINDOW)/WINDOW, mode="valid")
        # plt.fill_between(np.arange(len(lo_ma)) + (WINDOW - 1), lo_ma, hi_ma, alpha=0.12)
    plt.title(f"Convergence (retour moyen ± IC) — multi-seeds")
    plt.xlabel("Épisodes"); plt.ylabel(f"Retour (MA {WINDOW})")
    plt.legend(ncol=3, fontsize=9)
    out1 = os.path.join(SAVE_DIR, "sensitivity_gamma_convergence_ci.png")
    plt.tight_layout(); plt.savefig(out1, dpi=150); plt.close()

    # -------- Fig 2 : Time-to-threshold vs γ --------
    gam = [g for g in GAMMAS]
    ttt_mean = [results[g]["ttt"].mean() for g in GAMMAS]
    ttt_std  = [results[g]["ttt"].std()  for g in GAMMAS]
    plt.figure(figsize=(8, 4.8))
    plt.bar([str(g) for g in gam], ttt_mean, yerr=ttt_std, capsize=4)
    plt.axhline(EPISODES, color='k', alpha=0.2, linestyle='--', linewidth=1)
    plt.title(f"Temps pour atteindre le seuil (MA {WINDOW} ≥ {THRESHOLD})")
    plt.xlabel("γ"); plt.ylabel("Épisodes (plus bas = plus rapide)")
    out2 = os.path.join(SAVE_DIR, "sensitivity_gamma_time_to_threshold.png")
    plt.tight_layout(); plt.savefig(out2, dpi=150); plt.close()

    # -------- Fig 3 : Performance finale (200 derniers) --------
    plt.figure(figsize=(8, 4.8))
    means, stds, labels = [], [], []
    K = 200
    for g in GAMMAS:
        tails = [c[-K:] for c in results[g]["ep_return"]]
        arr = np.stack(tails, axis=0).mean(axis=1)  # moyenne par seed sur les 200 derniers
        means.append(arr.mean()); stds.append(arr.std()); labels.append(f"{g:.2f}")
    plt.bar(labels, means, yerr=stds, capsize=4)
    plt.title(f"Performance finale (moyenne sur {K} derniers épisodes)")
    plt.xlabel("γ"); plt.ylabel("Retour moyen")
    out3 = os.path.join(SAVE_DIR, "sensitivity_gamma_final.png")
    plt.tight_layout(); plt.savefig(out3, dpi=150); plt.close()

    # -------- Fig 4 : Longueur d'épisode (MA WINDOW) par γ (seed=0 illustratif) --------
    plt.figure(figsize=(9, 5))
    for g in GAMMAS:
        L = results[g]["ep_length"][0]  # seed=0
        if len(L) >= WINDOW:
            Ls = np.convolve(L, np.ones(WINDOW)/WINDOW, mode="valid")
            x = np.arange(len(Ls)) + (WINDOW - 1)
            plt.plot(x, Ls, label=f"γ={g:.2f}", linewidth=1.6)
    plt.title(f"Longueur des épisodes (MA {WINDOW}) — seed=0")
    plt.xlabel("Épisodes"); plt.ylabel("Nombre de pas")
    plt.legend(ncol=3, fontsize=9)
    out4 = os.path.join(SAVE_DIR, "sensitivity_gamma_episode_length.png")
    plt.tight_layout(); plt.savefig(out4, dpi=150); plt.close()

    # -------- Fig 5 : Exploration vs Exploitation (ε planifié, informatif) --------
    eps = eps_schedule(eps0=1.0, decay=0.995, eps_min=0.01, episodes=EPISODES)
    plt.figure(figsize=(8, 5))
    plt.stackplot(np.arange(EPISODES),
                  [eps, 1.0 - eps],
                  labels=["Exploration (ε)", "Exploitation (1-ε)"],
                  alpha=0.85)
    plt.title("Exploration vs Exploitation (planning ε)")
    plt.xlabel("Épisodes"); plt.ylabel("Proportion")
    plt.legend(loc="upper right")
    out5 = os.path.join(SAVE_DIR, "sensitivity_gamma_explore_exploit.png")
    plt.tight_layout(); plt.savefig(out5, dpi=150); plt.close()

    # -------- Fig 6 : (NOUVEAU) Proportion greedy réelle — 11 sous-graphes (un par γ) --------
    # Moyenne inter-seeds + MA(WINDOW) par γ, affichée en sous-figure.
    rows, cols = 3, 4  # 12 places; on n'en utilise que 11
    fig, axes = plt.subplots(rows, cols, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, g in enumerate(GAMMAS):
        ax = axes[i]
        props = results[g]["prop_greedy"]  # liste de 5 vecteurs (par seed)
        mean, _, _ = mean_ci(props)
        # lissage MA
        y = np.convolve(mean, np.ones(WINDOW)/WINDOW, mode="valid") if len(mean) >= WINDOW else mean
        x = np.arange(len(y)) + (WINDOW - 1 if len(mean) >= WINDOW else 0)
        ax.plot(x, y, linewidth=1.8)
        ax.set_title(f"γ={g:.2f}")
        ax.grid(True, alpha=0.25)
        if i % cols == 0:
            ax.set_ylabel("Prop. greedy")
        if i // cols == rows - 1:
            ax.set_xlabel("Épisodes")
    # masquer le dernier subplot si inutilisé
    if len(GAMMAS) < rows*cols:
        for j in range(len(GAMMAS), rows*cols):
            fig.delaxes(axes[j])

    fig.suptitle(f"Proportion d’actions greedy (réelle) — moyenne inter-seeds, MA {WINDOW}", y=0.98)
    out6 = os.path.join(SAVE_DIR, "sensitivity_gamma_prop_greedy_subplots.png")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out6, dpi=150)
    plt.close(fig)

    print("Figures sauvegardées :")
    for p in (out1, out2, out3, out4, out5, out6):
        print(" -", os.path.abspath(p))


if __name__ == "__main__":
    run()
