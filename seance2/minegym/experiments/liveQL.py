# --- hack simple pour rendre les imports fonctionnels quand on lance le script directement ---
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio.v2 as imageio               # MP4
from PIL import Image                      # downscale frames

from envs.gridworld import GridEnv
from agents.agentQL import QLearningAgent
from utils.plotting import (
    ensure_dir,
    policy_and_value_fig,
    summary_dashboard,
    plot_visit_heatmap,
    plot_dominant_actions,
    plot_V_heatmap,
    plot_pi_policy,
)

# ----------------- Dossiers de sortie -----------------
# (choisis l’un ou l’autre)
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "liveQLgoalsfixed")
# SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "liveQLgoalsnotfixed")
ensure_dir(SAVE_DIR)

# >>> SAUVEGARDES (checkpoints)
CKPT_BASE = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
CKPT_DIR = os.path.join(CKPT_BASE, "ql")
os.makedirs(CKPT_DIR, exist_ok=True)

# Évaluations périodiques pour sauvegarder le "best"
EVAL_EVERY     = 50       # toutes les 50 itérations
EVAL_EPISODES  = 20       # moyenne sur 20 épisodes greedy


# ----------------- outils vidéo -----------------
def downscale_rgb(rgb: np.ndarray, scale: float) -> np.ndarray:
    """Redimensionne une frame RGB (HxWx3) pour réduire la taille du MP4."""
    if scale >= 0.999:
        return rgb
    h, w = rgb.shape[:2]
    nh, nw = max(1, int(h*scale)), max(1, int(w*scale))
    img = Image.fromarray(rgb)
    img = img.resize((nw, nh), resample=Image.BILINEAR)
    return np.asarray(img)

def canvas_to_rgb(fig):
    """
    Retourne un np.ndarray HxWx3 (RGB) à partir du canvas,
    compatible avec TkAgg / QtAgg / Agg.
    """
    fig.canvas.draw()
    try:
        renderer = fig.canvas.get_renderer()
        rgba = np.asarray(renderer.buffer_rgba())
        return rgba[:, :, :3]
    except Exception:
        pass

    if hasattr(fig.canvas, "tostring_rgb"):
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return buf.reshape(h, w, 3)

    if hasattr(fig.canvas, "tostring_argb"):
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        return buf[:, :, [1, 2, 3]]

    raise RuntimeError("Impossible d'extraire une image du canvas (backend non supporté).")


def run(animate: bool = True):
    # ----------- définition de l'env -----------
    rows, cols = 6, 8
    start = 0
    goals = [46]  # 1 goal
    obstacles = [9, 10, 11, 12, 20, 28, 36, 37]

    # Goal fixe :
    env = GridEnv(rows=rows, cols=cols, obstacles=obstacles, goals=goals, start=start, seed=123)
    # OU goal mobile :
    # env = GridEnv(rows=rows, cols=cols, obstacles=obstacles, goals=goals,
    #               start=start, seed=123, moving_goal=True, moving_mode="cyclic")  # "random" aussi

    agent = QLearningAgent(env, gamma=0.9, alpha=0.2, epsilon=1.0, epsilon_decay=0.995)

    EPISODES  = 200
    MAX_STEPS = 50
    best_score = -np.inf

    # ----------- buffers de logs (pour summary_dashboard) -----------
    rewards_hist = []                  # reward total par épisode (pour le live)
    step_state, step_action = [], []
    step_reward, step_epsilon = [], []
    step_greedy, step_episode = [], []
    ep_returns, ep_lengths = [], []

    # ----------- animation live + vidéo -----------
    if animate:
        plt.ion()
        cmap = ListedColormap(["#FFFFFF", "#000000", "#2ECC71", "#E74C3C"])  # libre, obs, goal, agent

        fig = plt.figure(figsize=(13, 6))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.3], height_ratios=[1, 1])
        ax_grid = fig.add_subplot(gs[:, 0])   # grille à gauche (2 lignes)
        ax_step = fig.add_subplot(gs[0, 1])   # reward cumulatif épisode courant
        ax_ep   = fig.add_subplot(gs[1, 1])   # reward total par épisode

        # ---- MP4 seulement (libx264) ----
        SAVE_MP4 = True
        FPS = 12
        FRAME_SKIP = 2       # 1 frame sur 2
        MAX_FRAMES = 1200    # ~100s max à 12fps
        SCALE = 0.85         # réduire un peu la résolution

        MP4_PATH = os.path.join(SAVE_DIR, "live_training.mp4")
        mp4_writer = imageio.get_writer(
            MP4_PATH, fps=FPS,
            codec="libx264", quality=6, macro_block_size=None,
            pixelformat="yuv420p"
        ) if SAVE_MP4 else None

        frame_count = 0
        # taille fixe & paires (H.264)
        fixed_h = None
        fixed_w = None

        def render(step, episode, rewards_hist, curr_cum_rewards, epsilon, agent_state=None):
            nonlocal frame_count, fixed_h, fixed_w

            # --- Grille ---
            ax_grid.clear()
            M = np.zeros((env.rows, env.cols), dtype=int)
            for obs in env.obstacles:
                r, c = divmod(obs, env.cols); M[r, c] = 1
            for g in env.goals:
                r, c = divmod(g, env.cols);  M[r, c] = 2
            if agent_state is None:
                agent_state = env.state
            ar, ac = divmod(agent_state, env.cols)
            M[ar, ac] = 3

            ax_grid.imshow(M, cmap=cmap, vmin=0, vmax=3)
            ax_grid.set_aspect('equal')

            # quadrillage visible case par case
            ax_grid.set_xticks(np.arange(-0.5, env.cols, 1), minor=True)
            ax_grid.set_yticks(np.arange(-0.5, env.rows, 1), minor=True)
            ax_grid.grid(which='minor', color='gray', linestyle='-', linewidth=0.6, alpha=0.6)

            ax_grid.set_xticks([]); ax_grid.set_yticks([])
            ax_grid.tick_params(which='both', length=0)
            ax_grid.set_title(f"Épisode {episode+1}/{EPISODES} | Step {step} | ε={epsilon:.3f}")

            # --- cumul reward épisode courant
            ax_step.clear()
            ax_step.plot(curr_cum_rewards, linewidth=2)
            ax_step.set_title("Épisode courant — cumul du reward")
            ax_step.set_xlabel("Step"); ax_step.set_ylabel("Reward cumulé")

            # --- historique rewards épisodes
            ax_ep.clear()
            ax_ep.plot(range(1, len(rewards_hist)+1), rewards_hist,
                       alpha=0.6, linewidth=1.8, label="Reward par épisode")
            if len(rewards_hist) >= 10:
                kernel = np.ones(10)/10.0
                smooth = np.convolve(rewards_hist, kernel, mode="valid")
                x_s = list(range(10, 10+len(smooth)))
                ax_ep.plot(x_s, smooth, linewidth=2.2, label="Moyenne mobile (10)")
            ax_ep.set_title("Historique — reward total/épisode")
            ax_ep.set_xlabel("Épisode"); ax_ep.set_ylabel("Reward total")
            ax_ep.legend()

            # Affichage + capture
            plt.pause(0.01)
            if mp4_writer is not None:
                if (frame_count % FRAME_SKIP) == 0 and frame_count < MAX_FRAMES:
                    try:
                        frame = canvas_to_rgb(fig)   # HxWx3 RGB
                        frame = downscale_rgb(frame, SCALE)

                        # Fixer la taille de référence à la 1re frame (rendre paires)
                        h, w = frame.shape[:2]
                        if fixed_h is None or fixed_w is None:
                            fixed_h = h - (h % 2)
                            fixed_w = w - (w % 2)

                        # Forcer tailles constantes + paires (crop si nécessaire)
                        h2 = min(h, fixed_h)
                        w2 = min(w, fixed_w)
                        frame = frame[:h2, :w2, :]
                        if (h2 % 2) != 0:
                            h2 -= 1
                            frame = frame[:h2, :, :]
                        if (w2 % 2) != 0:
                            w2 -= 1
                            frame = frame[:, :w2, :]

                        mp4_writer.append_data(frame)
                    except Exception as e:
                        print("[MP4] erreur lors de l’écriture d’une frame:", repr(e))
                frame_count += 1

    # ----------- boucle d’entraînement -----------
    for ep in range(EPISODES):
        s = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        curr_cum_rewards = []

        if animate:
            render(step, ep, rewards_hist, curr_cum_rewards, agent.epsilon, s)

        while not done and step < MAX_STEPS:
            # epsilon-greedy
            if np.random.rand() < agent.epsilon:
                a = np.random.randint(env.action_space)
            else:
                a = int(np.argmax(agent.Q[s]))

            # marquer si l'action est gourmande (par rapport à Q)
            is_greedy = (a == int(np.argmax(agent.Q[s])))

            # transition
            ns, r, done = env.step(a)

            # MàJ Q-learning
            td_target = r + (0 if done else agent.gamma * np.max(agent.Q[ns]))
            td_error = td_target - agent.Q[s, a]
            agent.Q[s, a] += agent.alpha * td_error

            # logs step
            step_state.append(s)
            step_action.append(a)
            step_reward.append(r)
            step_epsilon.append(agent.epsilon)
            step_greedy.append(1 if is_greedy else 0)
            step_episode.append(ep)

            # avancer
            s = ns
            total_reward += r
            step += 1
            curr_cum_rewards.append(total_reward)

            if animate:
                render(step, ep, rewards_hist, curr_cum_rewards, agent.epsilon, s)

        # fin épisode
        rewards_hist.append(total_reward)
        ep_returns.append(total_reward)
        ep_lengths.append(step)

        # décroissance epsilon
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # évaluation + best
        if (ep + 1) % EVAL_EVERY == 0:
            score = agent.evaluate_policy(episodes=EVAL_EPISODES, max_steps=MAX_STEPS)
            print(f"[Eval] ep={ep+1} | score_moyen={score:.2f} | best={best_score:.2f}")
            if score > best_score:
                best_score = score
                agent.save(os.path.join(CKPT_DIR, "best"))
                print("=> Nouveau BEST sauvegardé dans:", os.path.abspath(os.path.join(CKPT_DIR, "best")))

    # close writer si ouvert
    try:
        mp4_writer.close()
        print("MP4 sauvegardé :", os.path.abspath(MP4_PATH))
    except Exception:
        pass

    if animate:
        plt.ioff()
        fig.tight_layout()
        plt.show()

    # ----------- Dashboards & figures finales -----------
    V_star = np.max(agent.Q, axis=1)
    pi_star = np.argmax(agent.Q, axis=1)

    plot_V_heatmap(
        env, V_star,
        os.path.join(SAVE_DIR, "V_star_heatmap_annotated.png"),
        annotate=True, fmt=".2f", show_markers=True
    )
    plot_pi_policy(env, pi_star, os.path.join(SAVE_DIR, "pi_star_grid.png"))

    # logs réels pour summary_dashboard
    logs = {
        "V_star": V_star, "pi_star": pi_star,
        "step_state": np.asarray(step_state, dtype=int),
        "step_action": np.asarray(step_action, dtype=int),
        "step_reward": np.asarray(step_reward, dtype=float),
        "step_epsilon": np.asarray(step_epsilon, dtype=float),
        "step_greedy": np.asarray(step_greedy, dtype=int),
        "step_episode": np.asarray(step_episode, dtype=int),
        "ep_return": np.asarray(ep_returns, dtype=float),
        "ep_length": np.asarray(ep_lengths, dtype=int),
    }
    summary_dashboard(env, logs, SAVE_DIR)
    policy_and_value_fig(env, V_star, pi_star, os.path.join(SAVE_DIR, "policy_value.png"))
    plot_visit_heatmap(env, np.array(step_state, dtype=int), os.path.join(SAVE_DIR, "visits.png"))
    plot_dominant_actions(env,
                          np.array(step_state, dtype=int),
                          np.array(step_action, dtype=int),
                          os.path.join(SAVE_DIR, "dominant_actions.png"))

    # -------- Explore/Exploit: empirique vs théorique --------
    def greedy_prop_per_episode(step_greedy_arr, step_episode_arr, n_episodes):
        prop = np.full(n_episodes, np.nan, dtype=float)
        for k in range(n_episodes):
            m = (step_episode_arr == k)
            if np.any(m):
                prop[k] = step_greedy_arr[m].mean()
        return prop

    EP_DONE = len(ep_returns)
    theo_eps = np.maximum(1.0 * (0.995 ** np.arange(EP_DONE)), 0.01)
    emp_greedy = greedy_prop_per_episode(
        np.asarray(step_greedy, dtype=float),
        np.asarray(step_episode, dtype=int),
        EP_DONE
    )

    plt.figure(figsize=(8, 5))
    if EP_DONE >= 50 and np.isfinite(emp_greedy).any():
        kernel = np.ones(50) / 50.0
        ok = np.isfinite(emp_greedy)
        sm = emp_greedy.copy()
        sm[ok] = np.convolve(emp_greedy[ok], kernel, mode="same")
        plt.plot(sm, label="Taux d'action gourmande (mesuré)", lw=2)
    else:
        plt.plot(emp_greedy, label="Taux d'action gourmande (mesuré)", lw=2)

    plt.plot(1 - theo_eps, label="Exploitation (théorique = 1-ε)", lw=2, alpha=0.85)
    plt.title("Exploration vs Exploitation — empirique vs théorique")
    plt.xlabel("Épisodes"); plt.ylabel("Proportion")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "live_explore_exploit_empirical.png"), dpi=150)
    plt.close()

    # Sauvegarde du modèle final
    agent.save(os.path.join(CKPT_DIR, "final"))
    print("Modèle final sauvegardé dans:", os.path.abspath(os.path.join(CKPT_DIR, "final")))


if __name__ == "__main__":
    run(animate=True)
