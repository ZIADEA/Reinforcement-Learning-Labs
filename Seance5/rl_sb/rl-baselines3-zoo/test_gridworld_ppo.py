import os
import time
import argparse
from typing import Any, Optional, List

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

# IMPORTANT : importe ton package pour que les envs soient enregistrés
import gridworld_env

# Pour l'affichage graphique et l'enregistrement
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import io


# ------------------------------------------------------------------
# Utilitaires pour accéder à ton GridEnv interne
# ------------------------------------------------------------------


def extract_core(env: Any):
    """
    Remonte la chaîne de wrappers jusqu'à trouver un objet qui a l'attribut `core`.
    - Si c'est un VecEnv, on prend env.envs[0]
    - Ensuite on suit env.env, env.env.env, ... jusqu'à trouver `.core`
    """
    base = env

    # Cas VecEnv (DummyVecEnv, SubprocVecEnv, etc.)
    if hasattr(base, "envs"):
        base = base.envs[0]

    # Déshabille les wrappers gymnasium jusqu'à trouver `.core`
    while True:
        if hasattr(base, "core"):
            return base.core  # ton GridEnv
        if hasattr(base, "env"):
            base = base.env
        else:
            break

    raise AttributeError("Impossible de trouver l'attribut 'core' dans la chaîne de wrappers.")


def obs_to_state(obs) -> int:
    """
    Convertit l'observation en index d'état (int).
    - Pendant l'entraînement, PPO travaille avec un VecEnv -> obs de forme (n_envs,).
    - Ici on n'utilise qu'un seul env, donc obs sera un scalaire ou un array de taille 1.
    """
    if isinstance(obs, np.ndarray):
        if obs.shape == (1,):
            return int(obs[0])
        return int(obs.squeeze())
    return int(obs)


# ------------------------------------------------------------------
# Rendu graphique avec Matplotlib (fond blanc, agent rouge, goal vert)
# ------------------------------------------------------------------


def render_grid_gui(core, state: int, ax):
    """
    Affiche la grille dans une fenêtre matplotlib :
    - fond blanc
    - obstacles en noir
    - goals en vert (avec un 'G')
    - agent en rouge (avec un 'A')
    """
    rows = core.rows
    cols = core.cols
    goals = list(core.goals)
    obstacles = set(core.obstacles)

    # Image RGB : fond blanc
    img = np.ones((rows, cols, 3), dtype=float)

    # Obstacles : noir
    for o in obstacles:
        r, c = divmod(o, cols)
        img[r, c] = [0.0, 0.0, 0.0]

    # Goals : vert
    for g in goals:
        r, c = divmod(g, cols)
        img[r, c] = [0.0, 1.0, 0.0]

    # Agent : rouge
    r_a, c_a = divmod(state, cols)
    img[r_a, c_a] = [1.0, 0.0, 0.0]

    ax.clear()
    ax.imshow(img, interpolation="nearest")

    # Grille
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])

    # Texte "G" et "A" pour être 100% sûr de ce qu’on regarde
    for g in goals:
        r, c = divmod(g, cols)
        ax.text(
            c, r, "G",
            ha="center", va="center",
            fontsize=16, fontweight="bold",
        )

    ax.text(
        c_a, r_a, "A",
        ha="center", va="center",
        fontsize=16, fontweight="bold",
    )

    # Titre avec indices pour debug visuel
    ax.set_title(f"GridWorld | Agent={state} | Goals={goals}")



def capture_frame(fig) -> np.ndarray:
    """
    Capture la figure matplotlib en image (numpy array H x W x 3),
    en passant par un buffer PNG en mémoire (compatible avec TkAgg).
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = imageio.imread(buf)
    buf.close()
    return img


# ------------------------------------------------------------------
# Fonction de test d'un modèle PPO sur un env donné
# ------------------------------------------------------------------


def run_eval(
    model_path: str,
    env_id: str,
    n_episodes: int = 3,
    max_steps: int = 50,
    sleep: float = 0.3,
    deterministic: bool = True,
    record_path: Optional[str] = None,
):
    print("=" * 80)
    print(f"Chargement du modèle PPO depuis : {model_path}")
    print(f"Environnement : {env_id}")
    print("=" * 80)

    if not os.path.isfile(model_path):
        print(f"[ERREUR] Le fichier modèle n'existe pas : {model_path}")
        return

    # Charger le modèle
    model = PPO.load(model_path)

    # Créer l’environnement
    env = gym.make(env_id)

    # Extraire le GridEnv interne
    try:
        core = extract_core(env)
    except Exception as e:
        print(f"[AVERTISSEMENT] Impossible d'extraire `core` depuis l'env : {e}")
        print("Le modèle sera quand même exécuté, mais sans rendu graphique ni enregistrement.")
        core = None

    # Préparer la fenêtre matplotlib
    fig, ax = None, None
    frames: List[np.ndarray] = []

    if core is not None:
        plt.ion()
        fig, ax = plt.subplots()
        try:
            fig.canvas.manager.set_window_title(f"Live PPO - {env_id}")
        except Exception:
            pass

    # ----------------------------------------------------------------------
    # Épisodes de test
    # ----------------------------------------------------------------------
    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()
        done = False
        step = 0

        print(f"\n===== Episode {ep} sur {env_id} =====")

        if core is not None:
            state = obs_to_state(obs)
            render_grid_gui(core, state, ax)
            plt.draw()
            plt.pause(0.001)

            if record_path is not None:
                frames.append(capture_frame(fig))

        while not done and step < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if core is not None:
                state = obs_to_state(obs)
                render_grid_gui(core, state, ax)
                plt.draw()
                plt.pause(sleep)

                if record_path is not None:
                    frames.append(capture_frame(fig))

            print(f"Step {step} | action={action} | reward={reward:.2f} | done={done}"
                  f"| goals={core.goals} | agent_state={state}"
                  )

        print(f"Episode {ep} terminé en {step} steps.")

    env.close()

    # ----------------------------------------------------------------------
    # Sauvegarde du GIF
    # ----------------------------------------------------------------------
    if core is not None and record_path is not None and len(frames) > 0:
        out_dir = os.path.dirname(record_path)
        if out_dir != "":
            os.makedirs(out_dir, exist_ok=True)

        fps = int(1.0 / max(sleep, 0.01))
        imageio.mimsave(record_path, frames, fps=fps)
        print(f"\n[INFO] Animation sauvegardée dans : {record_path}")

    if core is not None and fig is not None:
        print("Simulation terminée. Ferme la fenêtre matplotlib pour continuer.")
        plt.ioff()
        plt.show()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["static", "moving", "both"],
        default="both",
        help="Choisir l'env à tester : static, moving ou both.",
    )
    args = parser.parse_args()

    ROOT_SAVE_DIR = r"C:\Users\DJERI\VSCODE\Programmation\python\RL-class-propre\rl_sb\gridworld_runs"

    MODEL_STATIC = "logs/ppo/GridWorldStatic-v0_1/best_model.zip"
    MODEL_MOVING = "logs/ppo/GridWorldMoving-v0_1/best_model.zip"

    SAVE_STATIC = os.path.join(ROOT_SAVE_DIR, "gridworld_static_live.gif")
    SAVE_MOVING = os.path.join(ROOT_SAVE_DIR, "gridworld_moving_live.gif")

    if args.mode in ["static", "both"]:
        run_eval(
            model_path=MODEL_STATIC,
            env_id="GridWorldStatic-v0",
            n_episodes=10,
            max_steps=50,
            sleep=0.30,
            deterministic=True,
            record_path=SAVE_STATIC,
        )

    if args.mode in ["moving", "both"]:
        run_eval(
            model_path=MODEL_MOVING,
            env_id="GridWorldMoving-v0",
            n_episodes=10,
            max_steps=50,
            sleep=0.30,
            deterministic=True,
            record_path=SAVE_MOVING,
        )
