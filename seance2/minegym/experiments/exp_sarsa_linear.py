# --- hack simple pour rendre les imports fonctionnels quand on lance le script directement ---
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from envs.gridworld import GridEnv
from agents.agentSARSA0_control_V_par_epslon import Sarsa0LinearAgent
from utils.plotting import ensure_dir, plot_pi_policy, plot_V_heatmap

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "exp_sarsa_linear")
ensure_dir(SAVE_DIR)

def phi_coords_bias(env):
    def phi(s: int) -> np.ndarray:
        r, c = divmod(s, env.cols)
        R = 0.0 if env.rows <= 1 else r / (env.rows - 1)
        C = 0.0 if env.cols <= 1 else c / (env.cols - 1)
        return np.array([R, C, 1.0], dtype=np.float32)
    return phi

def build_env() -> GridEnv:
    rows, cols = 6, 8
    start = 0
    goals = [rows * cols - 1]
    obstacles = []
    return GridEnv(rows=rows, cols=cols,
                   obstacles=obstacles, goals=goals, start=start, seed=123,
                   reward_step=-1, reward_goal=50,
                   moving_goal=False)

def moving_average(x, w=50):
    if len(x) < w: 
        return np.array(x, dtype=float)
    kernel = np.ones(w)/w
    return np.convolve(x, kernel, mode="valid")

def run():
    env = build_env()
    phi = phi_coords_bias(env)

    agent = Sarsa0LinearAgent(
        env, phi=phi,
        gamma=0.99, alpha=0.1,
        epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
        episodes=2000, seed=0
    )

    ep_return = agent.train()

    # Courbe retour par épisode + MA(50)
    plt.figure(figsize=(8, 5))
    plt.plot(ep_return, lw=1.3, alpha=0.7, label="brut")
    ma = moving_average(ep_return, w=50)
    if len(ma) > 0:
        x = np.arange(len(ma)) + 49
        plt.plot(x, ma, lw=2.2, label="MA(50)")
    plt.title("SARSA(0) linéaire — retour par épisode")
    plt.xlabel("Épisodes"); plt.ylabel("Retour")
    plt.legend(); plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "sarsa_returns.png"), dpi=150)
    plt.close()

    # Politique & V* approximatif (depuis Q(s,a) ~ w_a^T φ(s))
    Qtab = agent.q_values_table()
    pi_star = np.argmax(Qtab, axis=1)
    V_star  = np.max(Qtab, axis=1)

    plot_pi_policy(env, pi_star, os.path.join(SAVE_DIR, "sarsa_pi_grid.png"))
    plot_V_heatmap(env, V_star, os.path.join(SAVE_DIR, "sarsa_V_heatmap.png"),
                   annotate=True, fmt=".2f", show_markers=True)

    print("Figures SARSA écrites dans :", os.path.abspath(SAVE_DIR))

if __name__ == "__main__":
    run()
