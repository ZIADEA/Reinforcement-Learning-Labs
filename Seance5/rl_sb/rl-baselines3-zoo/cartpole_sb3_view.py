# cartpole_sb3_view.py
# Usage (entraînement + visualisation) :
#   python cartpole_sb3_view.py --algo ppo --timesteps 10000 --episodes 5
# Pour juste rejouer (si un modèle existe déjà) :
#   python cartpole_sb3_view.py --algo ppo --episodes 5 --no-train

import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor

ALGOS = {
    "ppo": PPO,
    "dqn": DQN,
    "a2c": A2C,
}

def make_env(render: bool = True):
    # render_mode="human" ouvre une fenêtre (pygame requis)
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    # Monitor pour loguer les rewards/ep length
    env = Monitor(env)
    return env

def train_or_load(algo_name: str, timesteps: int, model_dir: Path, do_train: bool):
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "CartPole-v1.zip"

    Algo = ALGOS[algo_name]
    if model_path.exists() and not do_train:
        print(f"[INFO] Chargement du modèle existant: {model_path}")
        model = Algo.load(str(model_path))
        return model

    print(f"[INFO] Entraînement ({algo_name.upper()}) sur CartPole-v1 pendant {timesteps} étapes...")
    env = make_env(render=False)  # pas besoin d'afficher pendant l'entraînement
    # Policy multilayer perceptron, paramètres simples et robustes
    model = Algo("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(str(model_path))
    env.close()
    print(f"[INFO] Modèle sauvegardé dans: {model_path}")
    return model

def enjoy(model, episodes: int, fps: int = 60):
    # Fenêtre d’affichage
    env = make_env(render=True)
    # SB3 accepte un env non vectorisé (il vectorisera automatiquement si nécessaire)
    # On boucle sur quelques épisodes et on rend à ~fps
    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done, trunc = False, False
        ep_rew = 0.0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            ep_rew += reward
            # Gymnasium en mode "human" rend déjà à chaque step/reset,
            # mais on peut appeler env.render() pour forcer l’update.
            env.render()
            time.sleep(1.0 / fps)
        print(f"[INFO] Episode {ep}: reward={ep_rew:.1f}")
    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=ALGOS.keys(), default="ppo",
                        help="Algorithme: ppo | dqn | a2c")
    parser.add_argument("--timesteps", type=int, default=10000,
                        help="Nombre d’étapes d'entraînement si --no-train n’est pas passé")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Nombre d’épisodes à visualiser")
    parser.add_argument("--no-train", action="store_true",
                        help="Ne pas entraîner; juste charger et visualiser si modèle dispo")
    parser.add_argument("--models-dir", type=str, default="models",
                        help="Dossier où sauvegarder/charger les modèles")
    args = parser.parse_args()

    models_root = Path(args.models_dir) / args.algo / "CartPole-v1"
    model = train_or_load(args.algo, args.timesteps, models_root, do_train=not args.no_train)
    enjoy(model, episodes=args.episodes, fps=60)

if __name__ == "__main__":
    main()
