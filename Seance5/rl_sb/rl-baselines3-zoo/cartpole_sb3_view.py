# cartpole_sb3_view.py
# Entraînement + visualisation + GIF
#
# Exemple :
#   python cartpole_sb3_view.py --algo ppo --timesteps 10000 --episodes 5
# Juste rejouer (si le modèle existe déjà) :
#   python cartpole_sb3_view.py --algo ppo --episodes 5 --no-train

import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
import imageio.v2 as imageio

# ---------------------------------------------------------------------
# Répertoires fixes demandés
# ---------------------------------------------------------------------
BASE_ROOT = Path(r"C:\Users\DJERI\VSCODE\Programmation\python\RL-class-propre\rl_sb")
RLB_ROOT = BASE_ROOT / "rl-baselines3-zoo"
LOGS_ROOT = RLB_ROOT / "logs"
PPO_ROOT = LOGS_ROOT / "ppo"
GIF_ROOT = BASE_ROOT / "gridworld_runs"

# Tout ce qu’on sauvegarde est préfixé par "test"
MODEL_PATH = PPO_ROOT / "test_ppo_CartPole-v1.zip"
TB_LOG_DIR = LOGS_ROOT / "test_cartpole_tb"
GIF_PATH = GIF_ROOT / "test_cartpole.gif"

ALGOS = {
    "ppo": PPO,
    "dqn": DQN,
    "a2c": A2C,
}


def make_env(render: bool = True):
    """
    Environnement CartPole avec Monitor.
    - render=True : fenêtre (human)
    - render=False : pas de fenêtre
    """
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    env = Monitor(env)
    return env


def train_or_load(algo_name: str, timesteps: int, do_train: bool):
    """
    Entraîne ou charge un modèle.
    - Modèle sauvegardé dans MODEL_PATH
    - Logs TensorBoard dans TB_LOG_DIR
    """
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    TB_LOG_DIR.mkdir(parents=True, exist_ok=True)

    Algo = ALGOS[algo_name]

    if MODEL_PATH.exists() and not do_train:
        print(f"[INFO] Chargement du modèle existant: {MODEL_PATH}")
        model = Algo.load(str(MODEL_PATH))
        return model

    print(f"[INFO] Entraînement ({algo_name.upper()}) sur CartPole-v1 pendant {timesteps} étapes...")
    env = make_env(render=False)

    # TensorBoard log dans TB_LOG_DIR
    model = Algo(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(TB_LOG_DIR)
    )
    model.learn(total_timesteps=timesteps)
    model.save(str(MODEL_PATH))
    env.close()
    print(f"[INFO] Modèle sauvegardé dans: {MODEL_PATH}")
    print(f"[INFO] Logs TensorBoard dans: {TB_LOG_DIR}")
    return model


def enjoy(model, episodes: int, fps: int = 60):
    """
    Visualisation live dans une fenêtre (render_mode='human').
    """
    env = make_env(render=True)

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done, trunc = False, False
        ep_rew = 0.0

        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            ep_rew += reward
            env.render()
            time.sleep(1.0 / fps)

        print(f"[INFO] Episode {ep}: reward={ep_rew:.1f}")

    env.close()


def record_gif(model, episodes: int, fps: int = 30):
    """
    Rejoue quelques épisodes en mode 'rgb_array' et enregistre un GIF
    dans GIF_PATH, préfixé par 'test'.
    """
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    frames = []

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done, trunc = False, False

        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)

            frame = env.render()  # ndarray (H, W, 3)
            frames.append(frame)

    env.close()

    if frames:
        imageio.mimsave(GIF_PATH, frames, fps=fps)
        print(f"[INFO] GIF sauvegardé dans: {GIF_PATH}")
    else:
        print("[AVERTISSEMENT] Aucun frame capturé, GIF non créé.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        choices=ALGOS.keys(),
        default="ppo",
        help="Algorithme: ppo | dqn | a2c",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10000,
        help="Nombre d’étapes d'entraînement si --no-train n’est pas passé",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Nombre d’épisodes à visualiser / rejouer",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Ne pas entraîner; juste charger et visualiser si modèle dispo",
    )
    args = parser.parse_args()

    model = train_or_load(args.algo, args.timesteps, do_train=not args.no_train)

    # 1) Visualisation live dans une fenêtre
    enjoy(model, episodes=args.episodes, fps=60)

    # 2) Rejeu silencieux en rgb_array + enregistrement GIF
    record_gif(model, episodes=args.episodes, fps=30)


if __name__ == "__main__":
    main()
