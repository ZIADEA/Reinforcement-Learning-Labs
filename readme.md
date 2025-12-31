# Reinforcement Learning & Deep RL Labs — Portfolio 2025

> Collection of the homework, labs, and experiments that document everything I built for the Reinforcement Learning and Deep Reinforcement Learning course (Winter 2025). This repository is organized by séances so that the professor can follow the story of the course, re-run the key experiments, and inspect the analysis artifacts that accompany each delivery.

## Highlights

- **Structured history**: each `seance` folder contains the code, experiment scripts, figures, and results described in the accompanying README.
- **Core techniques covered**: from classical dynamic programming, Monte Carlo, and Q-Learning (seance1) to DQN variants (seance2 & seance4) and PPO experiments with Stable Baselines3 (Seance5).
- **Visual storytelling**: GIFs/figures live in `Sceance4/minegym/figures` and `Seance5/rl_sb/gridworld_runs` so you can inspect agent behaviours without running anything.

## Quick start

1. **Activate the Python environment.**

```
& C:\Users\DJERI\VSCODE\Programmation\python\environnements\rl_venv\Scripts\Activate.ps1
```

2. **Navigate to the session you want to inspect.**

- `cd seance1` for the foundational deliverables (MC, DP, PI, VI, Q-Learning agents).
- `cd seance2` or `cd Sceance4/minegym` for the GridWorld experiments and DQN implementations.
- `cd Seance5/rl_sb` for the PPO + Stable Baselines3 study.

3. **Re-run signature experiments.**

- `python -m minegym.experiments.liveQL`
- `python -m minegym.experiments.sensitivity_gammaQL`
- `python -m minegym.experiments.sensitivity_grid_sizeQL`
- `cd Seance5/rl_sb` and execute the `rl-baselines3-zoo` training or visualization scripts listed in `Seance5/readme.md`.

## Sessions at a glance

| Session | Focus | Key artifacts |
| --- | --- | --- |
| [seance1](seance1) | Basic reinforcement learning building blocks (Monte Carlo, dynamic programming, policy iteration, value iteration, tabular Q-Learning). | Deliverables such as `seance1/Liverable_2_DJERI-ALASSANI_OUBENOUPOU/agentQL.py` plus the entry scripts. |
| [seance2](seance2/minegym) | Parameterizable GridWorld + Q-Learning diagnostics (`liveQL`, `sensitivity_gammaQL`, `sensitivity_grid_sizeQL`). | See `seance2/minegym/README.md` for walkthroughs, reward conventions, and the corrected update with `w`. |
| [Sceance4](Sceance4/minegym) | Flexible GridWorld, corrected Q-Learning, and naïve vs DQN comparisons. | `Sceance4/minegym/README.md` and `Sceance4/minegym/DQNReadme.md` describe the experiments, CLI options, and plots. |
| [Seance5](Seance5) | PPO study with Stable-Baselines3 + rl-baselines3-zoo, covering static/moving GridWorld and CartPole. | Inspect `Seance5/readme.md`, the `rl_sb` folder, and `Seance5/rl_sb/gridworld_runs`. |
| [secance3](secance3/reinforcement) | Pacman-inspired project, autograder, and helper tooling for larger environments. | Files such as `secance3/learningAgents.py` and `secance3/reinforcementTestClasses.py`. |

## Visual assets and logs

- Browse `Sceance4/minegym/figures` for heatmaps, dashboards, and exploration/exploitation diagnostics produced during the DQN/QL comparisons.
- Explore `Seance5/rl_sb/gridworld_runs` for GIFs of the static and moving GridWorld agents plus the CartPole PPO agent.
- TensorBoard logs and checkpoints live under `Seance5/rl_sb/rl-baselines3-zoo/logs` for the PPO experiments.

## How to explore results

1. Start in the session folder you care about and read its README (see the links above) — each README now includes a TL;DR, quick commands, and the story behind the figures.
2. Inspect the `figures/`, `logs/`, or `checkpoints/` subfolders referenced by that session README.
3. Re-run the provided scripts if you want fresh figures or to compare new parameter settings.

Let me know if you would like a narrated tour of a specific session, new visual summaries, or additional metrics added to any experiment.

