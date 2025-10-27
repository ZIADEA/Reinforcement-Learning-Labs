# experiments/exp_flexible.py
# Backend non interactif par défaut (rapide)
import matplotlib
matplotlib.use("Agg")

import os, csv, argparse, sys
import numpy as np
import matplotlib.pyplot as plt

# hack import relatif si exécuté en script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.gridworld import GridEnv
from agents.agentDQN_flexible import FlexibleDQNConfig, FlexibleDQNAgent

# plotting utils (relatif puis absolu)
try:
    from ..utils.plotting import (
        ensure_dir,
        policy_and_value_fig,
        summary_dashboard,
        plot_visit_heatmap,
        plot_dominant_actions,
        plot_V_heatmap,
        plot_pi_policy,
        plot_training_curves,
    )
except Exception:
    from utils.plotting import (
        ensure_dir,
        policy_and_value_fig,
        summary_dashboard,
        plot_visit_heatmap,
        plot_dominant_actions,
        plot_V_heatmap,
        plot_pi_policy,
        plot_training_curves,
    )

def sample_new_goal_for_episode(env: GridEnv):
    forbidden = set(getattr(env, "obstacles", []))
    forbidden.add(env.start)
    env.goals = [env._sample_free_cell(forbidden)]

def eval_greedy(agent: FlexibleDQNAgent, env: GridEnv, episodes=10, max_steps=100):
    eps_bak = agent.epsilon
    agent.epsilon = 0.0
    tot = 0.0
    for _ in range(episodes):
        s = env.reset()
        done = False
        steps = 0
        rew = 0.0
        while not done and steps < max_steps:
            a = int(np.argmax(agent.q_values(s)))
            sn, r, done = env.step(a)
            s = sn; rew += r; steps += 1
        tot += rew
    agent.epsilon = eps_bak
    return tot / episodes

def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())

def main():
    p = argparse.ArgumentParser(description="Expérience flexible: NAIVE ou DQN")
    p.add_argument("--mode", choices=["naive", "dqn"], default="naive",
                   help="naive: pas de replay/target ; dqn: replay + target")
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--hidden", type=str, default="", help="Ex: '' (linéaire) ou '64,64'")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=8000)
    p.add_argument("--buffer", type=int, default=50000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--train-every", type=int, default=1)
    p.add_argument("--target-update-steps", type=int, default=5000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=20)

    # NEW: choix de loss
    p.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"],
                   help="Fonction de perte: 'mse' ou 'huber' (SmoothL1Loss).")
    p.add_argument("--huber-beta", type=float, default=1.0,
                   help="Delta (beta) pour la Huber (SmoothL1Loss).")

    args = p.parse_args()

    # Parsing robuste de --hidden
    raw = (args.hidden or "").strip().lower()
    if raw in ("", "none", "lin", "linear", "0"):
        hidden = ()
    else:
        hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip() != "")

    # Dossiers
    base_fig = args.save_dir or os.path.join(
        os.path.dirname(__file__), "..", "figures",
        f"flex_{args.mode}_{'lin' if len(hidden)==0 else 'mlp'}"
    )
    ensure_dir(base_fig)
    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints", f"flex_{args.mode}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Env 6x6, 1 goal par épisode, pas d'obstacles
    env = GridEnv(rows=args.rows, cols=args.cols, obstacles=None, goals=None,
                  start=None, seed=args.seed, moving_goal=False)
    n_states, n_actions = env.n_states, env.action_space

    # Flags de mode
    if args.mode == "naive":
        use_replay = False
        use_target = False
        # LR plus haut raisonnable en naïf linéaire
        lr = max(args.lr, 5e-2) if len(hidden)==0 else args.lr
    else:
        use_replay = True
        use_target = True
        lr = args.lr

    cfg = FlexibleDQNConfig(
        n_states=n_states, n_actions=n_actions,
        hidden=hidden, lr=lr, gamma=args.gamma, gradient_clip=10.0,
        # Loss configurable
        loss=args.loss, huber_beta=args.huber_beta,
        # Exploration
        epsilon_start=args.eps_start, epsilon_end=args.eps_end, epsilon_decay_steps=args.eps_decay_steps,
        # Replay/Target
        use_replay=use_replay, buffer_capacity=args.buffer, batch_size=args.batch,
        warmup_size=args.warmup, train_every=args.train_every,
        use_target=use_target, target_update_every_steps=args.target_update_steps,
        # Divers
        seed=args.seed, device=args.device
    )
    agent = FlexibleDQNAgent(cfg)

    # Affichage des paramètres du réseau
    n_params_online = _count_params(agent.online)
    print(f"[INFO] Réseau online: {'linéaire' if len(hidden)==0 else f'MLP {hidden}'} | "
          f"Paramètres: {n_params_online}")
    if agent.target is not None:
        n_params_target = _count_params(agent.target)
        print(f"[INFO] Réseau target: mêmes dimensions | Paramètres: {n_params_target}")
    print(f"[INFO] Loss: {args.loss} "
          f"{'(beta='+str(args.huber_beta)+')' if args.loss=='huber' else ''} | "
          f"Gamma: {args.gamma} | LR: {lr}")

    EPISODES  = args.episodes
    MAX_STEPS = args.max_steps
    best_score = -np.inf

    # Logs
    rewards_hist = []
    step_state, step_action = [], []
    step_reward, step_epsilon = [], []
    step_greedy, step_episode = [], []
    ep_returns, ep_lengths = [], []
    ep_losses, ep_eps, ep_theta_norm = [], [], []

    for ep in range(EPISODES):
        # goal fixé pendant l'épisode, mais resamplé à chaque épisode
        sample_new_goal_for_episode(env)
        s = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        ep_step_losses = []

        while not done and steps < MAX_STEPS:
            q_s = agent.q_values(s)
            a_greedy = int(np.argmax(q_s))
            a = agent.select_action(s)
            is_greedy = (a == a_greedy)

            sn, r, done = env.step(a)
            loss = agent.train_step(s, a, r, sn, done)
            if loss is not None and loss > 0.0:
                ep_step_losses.append(loss)

            # logs step
            step_state.append(s); step_action.append(a)
            step_reward.append(r); step_epsilon.append(agent.epsilon)
            step_greedy.append(1 if is_greedy else 0); step_episode.append(ep)

            s = sn; total_reward += r; steps += 1

        rewards_hist.append(total_reward)
        ep_returns.append(total_reward)
        ep_lengths.append(steps)
        ep_losses.append(float(np.mean(ep_step_losses)) if ep_step_losses else 0.0)
        ep_eps.append(agent.epsilon)
        ep_theta_norm.append(agent.params_l2())

        # logs terminal
        if (ep + 1) % 10 == 0:
            print(f"[{args.mode.upper()}|ep {ep+1:4d}/{EPISODES}] "
                  f"ret={total_reward:7.2f} steps={steps:3d} "
                  f"loss={ep_losses[-1]:.4f} eps={agent.epsilon:.3f} ||θ||={ep_theta_norm[-1]:.3f}")

        # évaluation périodique
        if (ep + 1) % args.eval_every == 0:
            score = eval_greedy(agent, env, episodes=args.eval_episodes, max_steps=MAX_STEPS)
            print(f"[Eval] ep={ep+1} | score_moyen={score:.2f} | best={best_score:.2f}")
            if score > best_score:
                best_score = score
                # sauvegarde des poids online
                import torch
                torch.save(agent.online.state_dict(), os.path.join(ckpt_dir, f"best_{args.mode}.pt"))
                print("=> Nouveau BEST sauvegardé:", os.path.abspath(os.path.join(ckpt_dir, f"best_{args.mode}.pt")))

    # --- Figures finales ---
    # Q(s,·) pour tous les états via le réseau online
    ALL_Q = np.zeros((env.n_states, env.action_space), dtype=float)
    for s in range(env.n_states):
        ALL_Q[s, :] = agent.q_values(s)
    V_star = ALL_Q.max(axis=1)
    pi_star = ALL_Q.argmax(axis=1)

    try:
        plot_V_heatmap(env, V_star, os.path.join(base_fig, "V_star_heatmap.png"),
                       annotate=True, fmt=".2f", show_markers=True)
        plot_pi_policy(env, pi_star, os.path.join(base_fig, "pi_star_grid.png"))
        policy_and_value_fig(env, V_star, pi_star, os.path.join(base_fig, "policy_value.png"))
        plot_visit_heatmap(env, np.array(step_state, dtype=int), os.path.join(base_fig, "visits.png"))
        plot_dominant_actions(env, np.array(step_state, dtype=int), np.array(step_action, dtype=int),
                              os.path.join(base_fig, "dominant_actions.png"))
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
        summary_dashboard(env, logs, base_fig, title_prefix=f"({args.mode.upper()})")
    except Exception as e:
        print("[plots finaux] optionnels non générés:", repr(e))

    # Plots d’historiques standardisés (loss, steps, epsilon & ||θ||, loss vs return)
    plot_training_curves(
        ep_return=np.asarray(ep_returns, dtype=float),
        ep_length=np.asarray(ep_lengths, dtype=int),
        ep_loss=np.asarray(ep_losses, dtype=float),
        ep_epsilon=np.asarray(ep_eps, dtype=float),
        ep_theta_norm=np.asarray(ep_theta_norm, dtype=float),
        save_dir=base_fig,
        prefix=f"{args.mode}_"
    )

    # CSV
    csv_path = os.path.join(base_fig, f"{args.mode}_logs.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode","return","length","mean_loss","epsilon","theta_norm"])
        for i in range(len(ep_returns)):
            w.writerow([i+1, ep_returns[i], ep_lengths[i], ep_losses[i], ep_eps[i], ep_theta_norm[i]])
    print("Logs CSV:", os.path.abspath(csv_path))

    # Poids finaux (online)
    import torch
    torch.save(agent.online.state_dict(), os.path.join(ckpt_dir, f"final_{args.mode}.pt"))
    print("Modèle final sauvegardé dans:", os.path.abspath(os.path.join(ckpt_dir, f"final_{args.mode}.pt")))

if __name__ == "__main__":
    main()
