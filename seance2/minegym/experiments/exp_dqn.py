# --- hack simple pour rendre les imports fonctionnels quand on lance le script directement ---
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from envs.gridworld import GridEnv
from agents.agentDQN import DQNAgent, make_phi_coords_bias
from utils.plotting import ensure_dir, plot_pi_policy, plot_V_heatmap

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "exp_dqn")
ensure_dir(SAVE_DIR)

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
    phi = make_phi_coords_bias(env.rows, env.cols)

    agent = DQNAgent(
        env,
        phi_fn=phi,
        hidden=64,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_episodes=1000,
        buffer_capacity=50_000,
        batch_size=64,
        target_update_every=200,
        seed=0,
    )

    logs = agent.train(episodes=2000, max_steps=250, warmup_steps=1000)
    ep_return = logs["ep_return"]

    # Courbe retour par épisode + MA(50)
    plt.figure(figsize=(8, 5))
    plt.plot(ep_return, lw=1.3, alpha=0.7, label="brut")
    ma = moving_average(ep_return, w=50)
    if len(ma) > 0:
        x = np.arange(len(ma)) + 49
        plt.plot(x, ma, lw=2.2, label="MA(50)")
    plt.title("DQN — retour par épisode")
    plt.xlabel("Épisodes"); plt.ylabel("Retour")
    plt.legend(); plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "dqn_returns.png"), dpi=150)
    plt.close()

    # Politique greedy & carte de valeurs approximative (depuis le réseau)
    import torch
    agent.qnet.eval()
    with torch.no_grad():
        Qtab = []
        for s in range(env.n_states):
            q = agent.qnet(agent._phi_t(s)).cpu().numpy()
            Qtab.append(q)
    Qtab = np.stack(Qtab, axis=0)
    pi_star = np.argmax(Qtab, axis=1)
    V_star  = np.max(Qtab, axis=1)

    plot_pi_policy(env, pi_star, os.path.join(SAVE_DIR, "dqn_pi_grid.png"))
    plot_V_heatmap(env, V_star, os.path.join(SAVE_DIR, "dqn_V_heatmap.png"),
                   annotate=True, fmt=".2f", show_markers=True)

    print("Figures DQN écrites dans :", os.path.abspath(SAVE_DIR))

if __name__ == "__main__":
    run()
