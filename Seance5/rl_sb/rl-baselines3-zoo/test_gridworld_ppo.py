import os
import time
from typing import Any

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO


# ------------------------------------------------------------------
# Utilitaires pour accéder à ton GridEnv interne et afficher la grille
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


def render_ascii(core, state: int):
    """
    Affiche une grille ASCII :
      A : agent
      G : goal
      X : obstacle
      . : case vide
    """
    rows = core.rows
    cols = core.cols
    goals = set(core.goals)
    obstacles = set(core.obstacles)

    lines = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            idx = r * cols + c
            if idx == state:
                row_cells.append("A")
            elif idx in goals:
                row_cells.append("G")
            elif idx in obstacles:
                row_cells.append("X")
            else:
                row_cells.append(".")
        lines.append(" ".join(row_cells))

    print("\n".join(lines))


def clear():
    """Nettoie l'écran de la console (Windows / Linux)."""
    os.system("cls" if os.name == "nt" else "clear")


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
):
    print("=" * 80)
    print(f"Chargement du modèle PPO depuis : {model_path}")
    print(f"Environnement : {env_id}")
    print("=" * 80)

    if not os.path.isfile(model_path):
        print(f"[ERREUR] Le fichier modèle n'existe pas : {model_path}")
        return

    # Chargement du modèle (CPU only par défaut)
    model = PPO.load(model_path)

    # Création d'un env gymnasium simple (pas de VecEnv ici)
    env = gym.make(env_id)

    # On récupère ton GridEnv interne pour avoir rows, cols, goals, obstacles
    try:
        core = extract_core(env)
    except Exception as e:
        print(f"[AVERTISSEMENT] Impossible d'extraire `core` depuis l'env : {e}")
        print("Le modèle sera quand même exécuté, mais sans rendu ASCII.")
        core = None

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()
        done = False
        step = 0

        print(f"\n===== Episode {ep} sur {env_id} =====")

        if core is not None:
            clear()
            print(f"===== Episode {ep} sur {env_id} | état initial =====")
            state = obs_to_state(obs)
            render_ascii(core, state)
            print()

        while not done and step < max_steps:
            # Action proposée par le PPO
            action, _ = model.predict(obs, deterministic=deterministic)

            # Step dans l'environnement
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated) or bool(truncated)
            step += 1

            if core is not None:
                clear()
                print(
                    f"===== Episode {ep} sur {env_id} | step {step}, "
                    f"reward {reward:.1f}, done={done} ====="
                )
                state = obs_to_state(obs)
                render_ascii(core, state)
                print()
                time.sleep(sleep)

        print(f"Episode {ep} terminé en {step} steps.")

    env.close()


# ------------------------------------------------------------------
# Main : tests statique (4x4) puis moving (4x4, si modèle présent)
# ------------------------------------------------------------------


if __name__ == "__main__":
    # 1) Modèle entraîné sur GridWorldStatic-v0 (ta taille actuelle = 4x4)
    MODEL_STATIC = "logs/ppo/GridWorldStatic-v0_1/GridWorldStatic-v0.zip"

    # 2) Modèle fine-tuné sur GridWorldMoving-v0 (après ton fine-tuning PPO)
    #    Adapte le chemin si le dossier a un suffixe différent (_2, etc.).
    MODEL_MOVING = "logs/ppo/GridWorldMoving-v0_1/GridWorldMoving-v0.zip"

    # Test du PPO sur l'environnement 4x4 avec goal fixe
    run_eval(
        model_path=MODEL_STATIC,
        env_id="GridWorldStatic-v0",
        n_episodes=3,
        max_steps=50,
        sleep=0.3,
        deterministic=True,
    )

    # Test du PPO sur l'environnement 4x4 avec goal moving (si le modèle existe)
    if os.path.isfile(MODEL_MOVING):
        run_eval(
            model_path=MODEL_MOVING,
            env_id="GridWorldMoving-v0",
            n_episodes=3,
            max_steps=50,
            sleep=0.3,
            deterministic=True,
        )
    else:
        print(
            "\n[INFO] Modèle fine-tuné pour GridWorldMoving-v0 introuvable : "
            f"{MODEL_MOVING}\n"
            "     → Lance d'abord le fine-tuning PPO, ou adapte le chemin dans ce script."
        )
