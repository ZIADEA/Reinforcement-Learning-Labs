# --- hack simple pour rendre les imports fonctionnels quand on lance le script directement ---
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from envs.gridworld import GridEnv
from agents.agentTD0predictionw import TD0LinearValuePrediction
from utils.plotting import ensure_dir, plot_V_heatmap, plot_pi_policy

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "exp_td0_linear")
ensure_dir(SAVE_DIR)

# ---- features φ(s) : coordonnées normalisées + biais (3D) ----
def phi_coords_bias(env):
    def phi(s: int) -> np.ndarray:
        r, c = divmod(s, env.cols)
        R = 0.0 if env.rows <= 1 else r / (env.rows - 1)
        C = 0.0 if env.cols <= 1 else c / (env.cols - 1)
        return np.array([R, C, 1.0], dtype=np.float32)
    return phi

# ---- politique fixe π (on-policy) : ici aléatoire uniforme pour la démo ----
def random_policy(env):
    rng = np.random.RandomState(0)
    def pi(s: int) -> int:
        return int(rng.randint(env.action_space))
    return pi

def build_env() -> GridEnv:
    rows, cols = 6, 8
    start = 0
    goals = [rows * cols - 1]
    obstacles = []
    return GridEnv(rows=rows, cols=cols,
                   obstacles=obstacles, goals=goals, start=start, seed=123,
                   reward_step=-1, reward_goal=50,
                   moving_goal=False)

def run():
    env = build_env()
    phi = phi_coords_bias(env)
    pi  = random_policy(env)   # remplace par une politique maison si tu veux

    agent = TD0LinearValuePrediction(
        env, phi=phi, policy=pi,
        alpha=0.1, gamma=0.99, episodes=1000
    )

    hist = agent.train()  # moyenne(V) par épisode

    # Courbe de convergence (moyenne de V^π)
    plt.figure(figsize=(7, 4))
    plt.plot(hist, lw=2)
    plt.title("TD(0) linéaire — Convergence de la moyenne de V^π")
    plt.xlabel("Épisodes"); plt.ylabel("moyenne(V)")
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "td0_convergence.png"), dpi=150)
    plt.close()

    # V estimé sous π, en heatmap
    V_map = np.array([agent.V(s) for s in range(env.n_states)], dtype=float)
    plot_V_heatmap(env, V_map, os.path.join(SAVE_DIR, "td0_Vpi_heatmap.png"),
                   annotate=True, fmt=".2f", show_markers=True)

    # --- Figure 3 : Politique suivie (la π fournie à TD0) ---
    pi_td = np.array([pi(s) for s in range(env.n_states)], dtype=int)
    plot_pi_policy(env, pi_td, os.path.join(SAVE_DIR, "td0_pi_followed.png"))

    # (Optionnel) sauver la politique
    np.save(os.path.join(SAVE_DIR, "td0_pi.npy"), pi_td)
    np.savetxt(os.path.join(SAVE_DIR, "td0_pi.csv"), pi_td, fmt="%d", delimiter=",")

    print("Figures TD(0) écrites dans :", os.path.abspath(SAVE_DIR))

if __name__ == "__main__":
    run()
