import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def moving_average(x: np.ndarray, w: int = 50) -> np.ndarray:
    if len(x) == 0:
        return x
    w = max(1, min(w, len(x)))
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[w:] - c[:-w]) / float(w)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def policy_and_value_fig(env, V_star: np.ndarray, pi_star: np.ndarray, save_path: str):
    """Heatmap de V* + flèches de la politique optimale."""
    grid_v = V_star.reshape(env.rows, env.cols)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(grid_v, cmap="viridis")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="V*(s)")

    # flèches
    for s, a in enumerate(pi_star):
        r, c = env.coords(s)
        if s == env.goals:
            ax.text(c, r, "G", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
            continue
        if s in env.obstacles:
            ax.text(c, r, "X", ha="center", va="center", color="red", fontsize=12, fontweight="bold")
            continue
        sym = {0:"←",1:"→",2:"↑",3:"↓"}[int(a)]
        ax.text(c, r, sym, ha="center", va="center", color="white", fontsize=12, fontweight="bold")

    ax.set_title("Valeur V* et politique optimale")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def summary_dashboard(env, logs: Dict, save_dir: str, title_prefix: str = ""):
    """Dashboard final demandé : distributions, moyennes, actions, exploration vs exploitation, etc."""
    ensure_dir(save_dir)
    # --- Données
    ep_return = logs["ep_return"]
    ep_length = logs["ep_length"]
    step_reward = logs["step_reward"]
    step_action = logs["step_action"]
    step_epsilon = logs["step_epsilon"]
    step_greedy = logs["step_greedy"]

    # --- 1) Reward distribution par action
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    ax = axes[0,0]
    data = [step_reward[step_action==a] for a in range(env.action_space)]
    ax.boxplot(data, labels=["←","→","↑","↓"], showfliers=False)
    ax.set_title("Distribution des récompenses par action")
    ax.set_ylabel("Récompense")

    # --- 2) Moyenne mobile du retour par épisode
    ax = axes[0,1]
    ax.plot(ep_return, alpha=0.3, label="Brut")
    ma = moving_average(ep_return, w=50)
    ax.plot(np.arange(len(ma))+50-1, ma, label="Lissé (50)")
    ax.set_title("Convergence (retour par épisode)")
    ax.set_xlabel("Épisodes"); ax.set_ylabel("Retour")
    ax.legend()

    # --- 3) Longueur d'épisode lissée
    ax = axes[0,2]
    ax.plot(ep_length, alpha=0.3, label="Brut")
    ma_len = moving_average(ep_length, w=50)
    ax.plot(np.arange(len(ma_len))+50-1, ma_len, label="Lissé (50)")
    ax.set_title("Longueur des épisodes (lissée)")
    ax.set_xlabel("Épisodes"); ax.set_ylabel("Nombre de pas")
    ax.legend()

    # --- 4) Distribution globale des actions
    ax = axes[1,0]
    counts = [np.sum(step_action==a) for a in range(env.action_space)]
    ax.bar(["←","→","↑","↓"], counts)
    ax.set_title("Distribution des actions")

    # --- 5) Taux 'action optimale' mesuré vs. courbe théorique exploration/exploitation
    ax = axes[1,1]
    # mesuré: proportion de pas où l'agent a joué 'greedy' au moment t
    greedy_rate = moving_average(step_greedy.astype(float), w=200)
    ax.plot(np.arange(len(greedy_rate))+200-1, greedy_rate, label="Taux d'action gourmande (mesuré)")
    # théorique (epsilon-greedy): P(exploit) = (1 - eps) + eps*(1/K)
    K = env.action_space
    eps_ma = moving_average(step_epsilon, w=200)
    theo = (1 - eps_ma) + eps_ma * (1.0 / K)
    ax.plot(np.arange(len(theo))+200-1, theo, label="Exploitation (théorique)")
    ax.set_title("Exploration vs Exploitation")
    ax.set_xlabel("Pas (lissé 200)"); ax.set_ylim(0,1.05)
    ax.legend()

    # --- 6) Récompense moyenne par pas (moyenne mobile)
    ax = axes[1,2]
    r_ma = moving_average(step_reward, w=200)
    ax.plot(np.arange(len(r_ma))+200-1, r_ma)
    ax.set_title("Récompense moyenne par pas (lissée)")
    ax.set_xlabel("Pas"); ax.set_ylabel("Récompense")

    fig.suptitle((title_prefix + " ").strip() + "Dashboard récapitulatif", y=1.02, fontsize=12)
    fig.tight_layout()
    out = os.path.join(save_dir, "summary_dashboard.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out

def plot_visit_heatmap(env, states: np.ndarray, save_path: str):
    counts = np.zeros(env.n_states, dtype=int)
    for s in states: counts[s]+=1
    grid = counts.reshape(env.rows, env.cols)
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(grid, cmap='Reds')
    plt.colorbar(im, ax=ax, label="Visites")
    ax.set_title("Heatmap des visites")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_dominant_actions(env, states: np.ndarray, actions: np.ndarray, save_path: str):
    mat = np.zeros((env.n_states, env.action_space), dtype=int)
    for s,a in zip(states, actions):
        mat[s,a]+=1
    dom = np.argmax(mat, axis=1)
    fig, ax = plt.subplots(figsize=(5,5))
    grid = np.max(mat, axis=1).reshape(env.rows, env.cols)
    im = ax.imshow(grid, cmap='Blues')
    plt.colorbar(im, ax=ax, label="Nb max action")
    for s,a in enumerate(dom):
        if s in env.obstacles: 
            r,c = env.coords(s)
            ax.text(c, r, "X", ha='center', va='center', color='red')
        elif s in env.goals:
            r,c = env.coords(s)
            ax.text(c, r, "G", ha='center', va='center', color='white')
        else:
            r,c = env.coords(s)
            ax.text(c, r, env.ACTIONS[a], ha='center', va='center', color='black')
    ax.set_title("Action dominante par case")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)



def plot_V_heatmap(env, V, save_path, annotate=False, fmt=".1f", show_markers=True):
    """
    Heatmap des valeurs V* avec option d'annotation des valeurs numériques.
    - annotate: si True, écrit la valeur exacte dans chaque case
    - fmt: format des nombres (ex: '.1f', '.2f')
    - show_markers: affiche '■' pour obstacles et 'G' pour goals
    """
    rows, cols = env.rows, env.cols
    V_grid = np.array(V).reshape(rows, cols)

    # bornes pour avoir un contraste stable
    vmin, vmax = np.nanmin(V_grid), np.nanmax(V_grid)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(V_grid, cmap="viridis", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("V*(s)")
    ax.set_title("V* (valeur optimale par état)")
    ax.set_xticks([]); ax.set_yticks([])

    # Marqueurs obstacles / goals (optionnels)
    if show_markers:
        goals = set(getattr(env, "goals", [])) if hasattr(env, "goals") else {getattr(env, "goal")}
        for obs in getattr(env, "obstacles", []):
            r, c = divmod(obs, cols)
            ax.text(c, r, "■", ha="center", va="center",
                    color="black", fontsize=12, fontweight="bold")
        for g in goals:
            r, c = divmod(g, cols)
            ax.text(c, r, "G", ha="center", va="center",
                    color="limegreen", fontsize=12, fontweight="bold")

    # Annotation des valeurs
    if annotate:
        # Couleur de texte en fonction de la luminance locale pour rester lisible
        for r in range(rows):
            for c in range(cols):
                v = V_grid[r, c]
                # normalisation via le colormap pour savoir si fond est sombre/clair
                luminance = im.norm(v)  # ~0: sombre; ~1: clair (avec 'viridis')
                txt_color = "white" if luminance < 0.5 else "black"
                ax.text(c, r, format(v, fmt),
                        ha="center", va="center",
                        fontsize=9, color=txt_color)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_pi_policy(env, pi, save_path):
    """Affiche et sauvegarde la politique optimale π* avec flèches."""
    rows, cols = env.rows, env.cols
    arrow_map = {0:"←", 1:"→", 2:"↑", 3:"↓"}
    pi_grid = np.array(pi).reshape(rows, cols)
    plt.figure(figsize=(6, 5))
    plt.imshow(np.ones((rows, cols)) * 0.95, cmap="Greys", vmin=0, vmax=1)
    for r in range(rows):
        for c in range(cols):
            idx = r*cols + c
            if idx in env.obstacles:
                txt, color = "■", "black"
            elif idx in env.goals:
                txt, color = "G", "green"
            else:
                txt, color = arrow_map.get(pi_grid[r, c], "?"), "black"
            plt.text(c, r, txt, ha='center', va='center', fontsize=14,
                     color=color, fontweight="bold")
    plt.title("Politique optimale π*")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()