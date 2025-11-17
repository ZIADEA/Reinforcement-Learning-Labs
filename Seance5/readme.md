# RL GridWorld + CartPole – PPO, Fine-Tuning, Convergence et Analyse des hyperparamètres

Ce dépôt correspond à une mini-étude expérimentale autour de PPO appliqué à :

1. Un environnement **GridWorld statique** (`GridWorldStatic-v0`, goal fixe)
2. Un environnement **GridWorld avec goal mobile** (`GridWorldMoving-v0`)
3. Un cas de **transfert raté** (finetuning du Moving à partir d'un agent Static déjà entraîné)
4. Un environnement de référence **CartPole-v1**

L'implémentation repose sur :

- **Stable-Baselines3**
- **rl-baselines3-zoo**
- Un environnement GridWorld custom inspiré de la séance 2 du dépôt principal :  
  https://github.com/ZIADEA/Reinforcement-Learning-Labs/tree/main/seance2/minegym

Une partie importante de l'analyse est faite via **TensorBoard** (convergence des récompenses, longueurs d'épisodes, pertes) et des **GIF** de visualisation qualitative.

Le projet présenté ici se trouve dans :  
https://github.com/ZIADEA/Reinforcement-Learning-Labs/tree/main/Seance5/rl_sb

---

## 1. Structure et fichiers importants

### 1.1. Environnement GridWorld custom

Chemins locaux (dans ce dépôt) et liens GitHub correspondants :

- `gridworld_env/gridworld_env/grid_core.py`  
  Logique interne de la grille (indexation, déplacements, gestion des goals, récompenses).  
  GitHub :  
  https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/rl_sb/gridworld_env/gridworld_env/grid_core.py

- `gridworld_env/gridworld_env/grid_sb3_env.py`  
  Wrapper Gymnasium / SB3 :  
  - enregistre les environnements `GridWorldStatic-v0` et `GridWorldMoving-v0`  
  - implémente `reset()` / `step()` compatibles avec **rl-baselines3-zoo**  
  - convertit l'état de la grille en observation entière.  
  GitHub :  
  https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/rl_sb/gridworld_env/gridworld_env/grid_sb3_env.py

- `gridworld_env/gridworld_env/__init__.py`  
  Point d'entrée du package `gridworld_env`.  
  GitHub :  
  https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/rl_sb/gridworld_env/gridworld_env/__init__.py

L'installation locale du package se fait avec :

```bash
cd gridworld_env
pip install -e .
```

---

### 1.2. Scripts d'entraînement et de visualisation

Tous ces fichiers se trouvent dans `rl-baselines3-zoo/` :

* `rl-baselines3-zoo/train.py`
  Script d'entraînement générique du zoo (appelé via `python -m rl_zoo3.train`).
  GitHub :
  [https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/rl_sb/rl-baselines3-zoo/train.py](https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/rl_sb/rl-baselines3-zoo/train.py)

* `rl-baselines3-zoo/test_gridworld_ppo.py`
  Script maison pour GridWorld :
  * charge un modèle PPO sur `GridWorldStatic-v0` ou `GridWorldMoving-v0`
  * affiche l'agent en live (Matplotlib, agent rouge, goal vert)
  * enregistre des GIF dans `gridworld_runs/`
  * logge dans le terminal : action, récompense, position agent, position goal.
    
  GitHub :
  [https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/rl_sb/rl-baselines3-zoo/test_gridworld_ppo.py](https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/rl_sb/rl-baselines3-zoo/test_gridworld_ppo.py)

* `rl-baselines3-zoo/cartpole_sb3_view.py`
  Script maison pour CartPole :
  * entraîne un PPO sur `CartPole-v1`
  * sauvegarde le modèle et les logs
  * rejoue quelques épisodes en mode `render="human"`
  * génère un GIF dans `gridworld_runs/test_cartpole.gif`.
    
  GitHub :
  [https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/rl_sb/rl-baselines3-zoo/cartpole_sb3_view.py](https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/rl_sb/rl-baselines3-zoo/cartpole_sb3_view.py)

---

### 1.3. Hyperparamètres PPO

Fichier principal (zoo) :

* `rl-baselines3-zoo/hyperparams/ppo.yml`
  GitHub :
  [https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/rl_sb/rl-baselines3-zoo/hyperparams/ppo.yml](https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/rl_sb/rl-baselines3-zoo/hyperparams/ppo.yml)

Extrait des entrées ajoutées/modifiées :

```yaml
GridWorldStatic-v0:
  n_timesteps: 100000
  policy: "MlpPolicy"
  n_steps: 256
  batch_size: 64
  n_epochs: 4
  gamma: 0.99
  learning_rate: 3.0e-4
  ent_coef: 0.0
  clip_range: 0.2
  vf_coef: 0.5
  max_grad_norm: 0.5

GridWorldMoving-v0:
  n_timesteps: 50000          # configuration de base, utilisée pour certains runs
  policy: "MlpPolicy"
  n_steps: 256
  batch_size: 64
  n_epochs: 4
  gamma: 0.99
  learning_rate: 1.0e-4       # LR plus petit pour faciliter l'adaptation
  ent_coef: 0.0
  clip_range: 0.2
  vf_coef: 0.5
  max_grad_norm: 0.5
```

Ces configurations sont automatiquement utilisées par `rl_zoo3.train` quand on passe `--env GridWorldStatic-v0` ou `--env GridWorldMoving-v0`.

---

### 1.4. Modèles PPO sauvegardés

Les modèles principaux se trouvent dans `rl-baselines3-zoo/logs/`.

#### Runs de base (100k + essais finetune)

* Static 100k (run de base) :
  `rl-baselines3-zoo/logs/ppo/GridWorldStatic-v0_1/`
  GitHub (dossier) :
  [https://github.com/ZIADEA/Reinforcement-Learning-Labs/tree/main/Seance5/rl_sb/rl-baselines3-zoo/logs/ppo/GridWorldStatic-v0_1](https://github.com/ZIADEA/Reinforcement-Learning-Labs/tree/main/Seance5/rl_sb/rl-baselines3-zoo/logs/ppo/GridWorldStatic-v0_1)

* Moving 100k (entraînement direct, PPO vierge) :
  `rl-baselines3-zoo/logs/ppo/GridWorldMoving-v0_1/`
  GitHub :
  [https://github.com/ZIADEA/Reinforcement-Learning-Labs/tree/main/Seance5/rl_sb/rl-baselines3-zoo/logs/ppo/GridWorldMoving-v0_1](https://github.com/ZIADEA/Reinforcement-Learning-Labs/tree/main/Seance5/rl_sb/rl-baselines3-zoo/logs/ppo/GridWorldMoving-v0_1)

* Moving finetuné à partir du Static (transfert raté) :
  `rl-baselines3-zoo/logs/ppo/GridWorldMoving-v0_1_fintune_sur_gridword_static/`
  GitHub :
  [https://github.com/ZIADEA/Reinforcement-Learning-Labs/tree/main/Seance5/rl_sb/rl-baselines3-zoo/logs/ppo/GridWorldMoving-v0_1_fintune_sur_gridword_static](https://github.com/ZIADEA/Reinforcement-Learning-Labs/tree/main/Seance5/rl_sb/rl-baselines3-zoo/logs/ppo/GridWorldMoving-v0_1_fintune_sur_gridword_static)

#### Runs dédiés à l'étude de la convergence (Static)

* Static 50k :
  `rl-baselines3-zoo/logs/ppo_static_50k/ppo/GridWorldStatic-v0_1/`
  TensorBoard : `rl-baselines3-zoo/logs/logs_tensorboard_ppo_gridworld_static_50k/`
  GitHub :
  * Modèle :
    [https://github.com/ZIADEA/Reinforcement-Learning-Labs/tree/main/Seance5/rl_sb/rl-baselines3-zoo/logs/ppo_static_50k](https://github.com/ZIADEA/Reinforcement-Learning-Labs/tree/main/Seance5/rl_sb/rl-baselines3-zoo/logs/ppo_static_50k)
  * Logs TB :
    [https://github.com/ZIADEA/Reinforcement-Learning-Labs/tree/main/Seance5/rl_sb/rl-baselines3-zoo/logs/logs_tensorboard_ppo_gridworld_static_50k](https://github.com/ZIADEA/Reinforcement-Learning-Labs/tree/main/Seance5/rl_sb/rl-baselines3-zoo/logs/logs_tensorboard_ppo_gridworld_static_50k)

* Static 400k :
  `rl-baselines3-zoo/logs/ppo_static_400k/ppo/GridWorldStatic-v0_1/`
  TensorBoard : `rl-baselines3-zoo/logs/logs_tensorboard_ppo_gridworld_static_400k/`
  GitHub :
  * Modèle :
    [https://github.com/ZIADEA/Reinforcement-Learning-Labs/tree/main/Seance5/rl_sb/rl-baselines3-zoo/logs/ppo_static_400k](https://github.com/ZIADEA/Reinforcement-Learning-Labs/tree/main/Seance5/rl_sb/rl-baselines3-zoo/logs/ppo_static_400k)
  * Logs TB :
    [https://github.com/ZIADEA/Reinforcement-Learning-Labs/tree/main/Seance5/rl_sb/rl-baselines3-zoo/logs/logs_tensorboard_ppo_gridworld_static_400k](https://github.com/ZIADEA/Reinforcement-Learning-Labs/tree/main/Seance5/rl_sb/rl-baselines3-zoo/logs/logs_tensorboard_ppo_gridworld_static_400k)

#### Runs dédiés à l'étude de la convergence (Moving)

* Moving 400k :
  `rl-baselines3-zoo/logs/ppo_moving_400k/ppo/GridWorldMoving-v0_1/`
  TensorBoard : `rl-baselines3-zoo/logs/logs_tensorboard_ppo_gridworld_moving_400k/`

* Moving 600k :
  `rl-baselines3-zoo/logs/ppo_moving_600k/ppo/GridWorldMoving-v0_1/`
  TensorBoard : `rl-baselines3-zoo/logs/logs_tensorboard_ppo_gridworld_moving_600k/`

* Moving long (run intermédiaire plus long, non détaillé ici) :
  `rl-baselines3-zoo/logs/ppo_moving_long/`
  TensorBoard : `rl-baselines3-zoo/logs/logs_tensorboard_ppo_gridworld_moving_long/`

* Moving long2 (run le plus long, environ 1.6M steps) :
  `rl-baselines3-zoo/logs/ppo_moving_long2/ppo/GridWorldMoving-v0_1/`
  TensorBoard : `rl-baselines3-zoo/logs/logs_tensorboard_ppo_gridworld_moving_long2/`

GitHub (dossier logs Moving) :
[https://github.com/ZIADEA/Reinforcement-Learning-Labs/tree/main/Seance5/rl_sb/rl-baselines3-zoo/logs](https://github.com/ZIADEA/Reinforcement-Learning-Labs/tree/main/Seance5/rl_sb/rl-baselines3-zoo/logs)

#### CartPole

* PPO CartPole-v1 (zoo) :
  `rl-baselines3-zoo/logs/ppo/CartPole-v1_1/`
* Copie globale du modèle :
  `models/ppo/CartPole-v1/CartPole-v1.zip`
  GitHub :
  [https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/rl_sb/models/ppo/CartPole-v1/CartPole-v1.zip](https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/rl_sb/models/ppo/CartPole-v1/CartPole-v1.zip)

---

### 1.5. GIF des politiques apprises

Tous les GIF sont dans `gridworld_runs/` et sont affichés ci-dessous :

#### GridWorld Static

**Agent Static – 100k steps**
![GridWorld static 100k](https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/rl_sb/gridworld_runs/gridworld_runs/gridworld_static_live.gif?raw=true)

**Agent Static – 50k steps**
![GridWorld static 50k](https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/rl_sb/gridworld_runs/gridworld_runs/gridworld_ppo_static_50k_live.gif?raw=true)

**Agent Static – 400k steps**
![GridWorld static 400k](https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/rl_sb/gridworld_runs/gridworld_runs/gridworld_ppo_static_400k_live.gif?raw=true)

#### GridWorld Moving

**Agent Moving – Finetune depuis Static (transfert raté)**
![GridWorld moving finetune](https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/rl_sb/gridworld_runs/gridworld_moving_live_fintune_sur_gridword_static.gif?raw=true)

**Agent Moving – 100k steps from scratch**
![GridWorld moving 100k](https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/rl_sb/gridworld_runs/gridworld_moving_live.gif?raw=true)

**Agent Moving – 400k steps**
![GridWorld moving 400k](https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/rl_sb/gridworld_runs/gridworld_ppo_moving_400k.gif?raw=true)

**Agent Moving – 600k steps**
![GridWorld moving 600k](https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/rl_sb/gridworld_runs/gridworld_ppo_moving_600k.gif?raw=true)

**Agent Moving – Long2 (≈1.6M steps)**
![GridWorld moving long2](https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/rl_sb/gridworld_runs/gridworld_movinglong2_live.gif?raw=true)

#### CartPole

**Agent CartPole-v1 – PPO**
![CartPole PPO](https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/rl_sb/gridworld_runs/test_cartpole.gif?raw=true)

---

## 2. Description des environnements GridWorld

### 2.1. GridWorldStatic-v0

Caractéristiques :

* Grille 4×4
* Un seul goal fixe
* L'agent commence dans une position déterminée (ou aléatoire suivant la config)
* Récompenses typiques :
  * −1 par step
  * +30 lorsque le goal est atteint
* Épisodes terminés à la collision avec le goal ou au dépassement d'un nombre maximal de steps.

Tâche simple : apprendre à aller rapidement vers un goal fixe.

### 2.2. GridWorldMoving-v0

Caractéristiques :

* Même grille 4×4
* Le goal change de position à chaque step ou suivant une dynamique aléatoire
* Même structure de récompense globale (pénalité par step, bonus à l'atteinte du goal)

Tâche plus difficile : l'agent doit **poursuivre une cible mobile** dans un environnement non stationnaire.

---

## 3. Commandes d'entraînement (PPO)

Toutes les commandes ci-dessous se lancent depuis :

```bash
cd rl-baselines3-zoo
```

### 3.1. GridWorldStatic-v0

#### Static – 50k steps

```bash
python -m rl_zoo3.train ^
  --algo ppo ^
  --env GridWorldStatic-v0 ^
  -n 50000 ^
  -f logs/ppo_static_50k ^
  --tensorboard-log logs/logs_tensorboard_ppo_gridworld_static_50k/
```

* Modèle et logs RL : `logs/ppo_static_50k/`
* Logs TensorBoard : `logs/logs_tensorboard_ppo_gridworld_static_50k/`

#### Static – 400k steps

```bash
python -m rl_zoo3.train ^
  --algo ppo ^
  --env GridWorldStatic-v0 ^
  -n 400000 ^
  -f logs/ppo_static_400k ^
  --tensorboard-log logs/logs_tensorboard_ppo_gridworld_static_400k/
```

* Modèle et logs RL : `logs/ppo_static_400k/`
* Logs TensorBoard : `logs/logs_tensorboard_ppo_gridworld_static_400k/`

---

### 3.2. GridWorldMoving-v0

#### Moving – 400k steps (from scratch)

```bash
python -m rl_zoo3.train ^
  --algo ppo ^
  --env GridWorldMoving-v0 ^
  -n 400000 ^
  -f logs/ppo_moving_400k ^
  --tensorboard-log logs/logs_tensorboard_ppo_gridworld_moving_400k/
```

#### Moving – 600k steps (from scratch)

```bash
python -m rl_zoo3.train ^
  --algo ppo ^
  --env GridWorldMoving-v0 ^
  -n 600000 ^
  -f logs/ppo_moving_600k ^
  --tensorboard-log logs/logs_tensorboard_ppo_gridworld_moving_600k/
```

#### Moving – run long (≈ 800k) et long2 (≈ 1.6M)

Même structure, seul `-n` et le préfixe de log changent. Par exemple pour le run très long (long2) :

```bash
python -m rl_zoo3.train ^
  --algo ppo ^
  --env GridWorldMoving-v0 ^
  -n 1600000 ^
  -f logs/ppo_moving_long2 ^
  --tensorboard-log logs/logs_tensorboard_ppo_gridworld_moving_long2/
```

---

### 3.3. Moving – finetune à partir du modèle Static (transfert raté)

Exemple de transfert qui se passe mal (on part d'un Static déjà convergé) :

```bash
python -m rl_zoo3.train ^
  --algo ppo ^
  --env GridWorldMoving-v0 ^
  -i logs/ppo/GridWorldStatic-v0_1/GridWorldStatic-v0.zip ^
  -n 50000 ^
  --tensorboard-log logs/logs_tensorboard_ppo_gridworld_moving_fintune_sur_gridword_static/
```

Les résultats sont dans :

* `logs/ppo/GridWorldMoving-v0_1_fintune_sur_gridword_static/`
* `logs/logs_tensorboard_ppo_gridworld_moving_fintune_sur_gridword_static/`

On observe que l'agent reste bloqué sur une stratégie adaptée au goal fixe et n'apprend pas réellement à suivre un goal mobile.

---

### 3.4. CartPole-v1

Script dédié :

```bash
cd rl-baselines3-zoo
python cartpole_sb3_view.py --algo ppo --timesteps 20000 --episodes 5
```

Résultats :

* Logs RL : `logs/ppo/CartPole-v1_1/`
* Logs TensorBoard : `logs/test_cartpole_tb/`
* Modèle global : `../../models/ppo/CartPole-v1/CartPole-v1.zip`
* GIF : `../../rl_sb/gridworld_runs/gridworld_runs/test_cartpole.gif`

---

## 4. Visualisation et GIF

Script principal :

```bash
cd rl-baselines3-zoo
python test_gridworld_ppo.py --mode static    # pour tester l'agent static
python test_gridworld_ppo.py --mode moving   # pour tester l'agent moving
```

Par défaut, le script charge :

* Pour `--mode static` : un modèle PPO sur `GridWorldStatic-v0`
* Pour `--mode moving` : un modèle PPO sur `GridWorldMoving-v0`

Les GIF générés sont dans `../rl_sb/gridworld_runs/gridworld_runs/` et sont affichés dans la section 1.5 ci-dessus.

---

## 5. TensorBoard : chemins et commandes

Depuis `rl-baselines3-zoo/` :

### Static

* 50k :
```bash
tensorboard --logdir logs/logs_tensorboard_ppo_gridworld_static_50k
```

* 400k :
```bash
tensorboard --logdir logs/logs_tensorboard_ppo_gridworld_static_400k
```

* Run de base Static (100k) :
```bash
tensorboard --logdir logs/logs_tensorboard_ppo_gridworld_static
```

### Moving

* 400k :
```bash
tensorboard --logdir logs/logs_tensorboard_ppo_gridworld_moving_400k
```

* 600k :
```bash
tensorboard --logdir logs/logs_tensorboard_ppo_gridworld_moving_600k
```

* Long (≈ 800k) :
```bash
tensorboard --logdir logs/logs_tensorboard_ppo_gridworld_moving_long
```

* Long2 (≈ 1.6M) :
```bash
tensorboard --logdir logs/logs_tensorboard_ppo_gridworld_moving_long2
```

### Moving finetune Static

```bash
tensorboard --logdir logs/logs_tensorboard_ppo_gridworld_moving_fintune_sur_gridword_static
```

### CartPole

```bash
tensorboard --logdir logs/test_cartpole_tb
```

---

## 6. Courbes de convergence (PNG)

Les figures suivantes ont été générées à partir des logs TensorBoard convertis en CSV, puis tracées en Python.
Elles sont stockées dans le dossier `images/` :

### GridWorld Static – Comparaison 50k vs 400k

**Récompense moyenne par épisode**
![Static ep_rew_mean](https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/images/static_50k_400k_ep_rew_mean.png?raw=true)

**Longueur moyenne d'épisode**
![Static ep_len_mean](https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/images/static_50k_400k_ep_len_mean.png?raw=true)

### GridWorld Moving – Comparaison 400k vs 600k vs 1.6M

**Récompense moyenne par épisode**
![Moving ep_rew_mean](https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/images/moving_400k_600k_1600k_ep_rew_mean.png?raw=true)

**Longueur moyenne d'épisode**
![Moving ep_len_mean](https://github.com/ZIADEA/Reinforcement-Learning-Labs/blob/main/Seance5/images/moving_400k_600k_1600k_ep_len_mean.png?raw=true)

---

## 7. Interprétation des résultats et estimation de n minimal

### 7.1. GridWorldStatic-v0

Observations sur les courbes (50k, 100k, 400k) :

* `rollout/ep_rew_mean` :
  * dès 50k steps, la récompense moyenne est déjà élevée et assez stable ;
  * passer à 100k puis 400k ne change presque pas la politique apprise : on gagne surtout en robustesse et en stabilité statistique.
* `rollout/ep_len_mean` :
  * les épisodes deviennent rapidement courts (l'agent atteint vite le goal) ;
  * les courbes se stabilisent rapidement.
* `train/value_loss` :
  * forte baisse au début, puis oscillations sur une plage basse dès 50k ;
  * les runs plus longs ne corrigent plus grand-chose, ils affinent un optimum déjà très bon.

Conclusion pour Static :

* La tâche est simple, l'agent trouve un bon comportement très vite.
* **50k steps sont déjà suffisants pour une politique raisonnable.**
* L'entraînement à 100k ou 400k sert surtout à lisser la variance des performances.

Résumé Static :

* **n minimal (politique utilisable)** : ≈ 50 000 steps
* **n recommandé (confort, convergence nette)** : entre 100 000 et 400 000 steps

---

### 7.2. GridWorldMoving-v0 (from scratch)

Sur les runs 400k, 600k et 1.6M (long2), on observe :

* À 400k :
  * `ep_rew_mean` progresse mais reste encore assez fluctuante ;
  * `ep_len_mean` est encore variable, les épisodes ne sont pas systématiquement courts ;
  * `value_loss` et `policy_gradient_loss` n'ont pas de vrai plateau stable.
* À 600k :
  * les courbes s'améliorent, la récompense moyenne est plus haute ;
  * la variance reste importante, les épisodes sont plus courts mais pas parfaitement stables ;
  * on voit que l'agent a appris une stratégie de poursuite, mais encore un peu fragile.
* À 1.6M (long2) :
  * `ep_rew_mean` atteint un plateau plus clair, avec des oscillations plus contenues ;
  * `ep_len_mean` se stabilise : l'agent atteint la cible mobile plus rapidement et de façon plus régulière ;
  * `value_loss` et `policy_gradient_loss` montrent un régime stationnaire raisonnable.

Conclusion pour Moving :

* L'environnement est non stationnaire (goal mobile), donc plus difficile.
* Les runs à 400k et même 600k montrent encore une **convergence incomplète**.
* Le run très long (≈ 1.6M) donne une image plus proche d'une convergence robuste.

Résumé Moving :

* **n minimal (politique correcte mais encore un peu fragile)** : ≈ 600 000 steps
* **n recommandé (convergence claire et comportement stable)** : ≈ 1 600 000 steps

---

### 7.3. Moving – finetune à partir de Static (transfert raté)

Sur le run finetune :

* `ep_rew_mean` reste assez basse et ne s'améliore presque pas ;
* l'agent adopte un comportement rigide et ne suit pas la cible mobile ;
* visuellement, le GIF montre que l'agent reste bloqué sur une case proche de l'ancienne position du goal fixe.

Interprétation :

* L'agent a appris une politique trop spécialisée sur le Static (goal fixe).
* Le finetune ne parvient pas à casser ce biais ; on reste dans un **mauvais optimum local**.
* Dans ce cas, il est plus efficace d'**entraîner l'agent Moving from scratch** plutôt que de partir d'un Static déjà convergé.

---

### 7.4. CartPole-v1

CartPole sert de référence de PPO sur une tâche classique :

* Convergence rapide de `ep_rew_mean` vers des valeurs proches du maximum ;
* `ep_len_mean` montre des épisodes très longs (souvent stoppés par la limite de temps) ;
* Comportement du GIF conforme : le pendule reste essentiellement en position verticale, avec des oscillations contrôlées du chariot.

---

## 8. Tableau récapitulatif des expériences

### 8.1. Entraînements et n recommandés

| Environnement      | Type d'expérience      | n (steps) typiques dans ce dépôt | Observations                          | n minimal raisonnable | n recommandé   |
| ------------------ | ---------------------- | -------------------------------- | ------------------------------------- | --------------------- | -------------- |
| GridWorldStatic-v0 | PPO from scratch       | 50k, 100k, 400k                  | Convergence très rapide, tâche simple | ≈ 50k                 | 100k–400k      |
| GridWorldMoving-v0 | PPO from scratch       | 100k, 400k, 600k, 1.6M           | Plus lent, non stationnaire           | ≈ 600k                | ≈ 1.6M         |
| GridWorldMoving-v0 | Finetune depuis Static | 50k                              | Transfert raté, politique biaisée     | Non recommandé        | Non recommandé |
| CartPole-v1        | PPO from scratch       | 20k (script dédié)               | Convergence très rapide               | ≈ 20k                 | 20k–50k        |

---

## 9. Reproduction rapide des expériences principales

Résumé minimal :

```bash
# 1. Se placer dans le dossier du zoo
cd rl-baselines3-zoo

# 2. Entraîner GridWorld Static – 50k
python -m rl_zoo3.train --algo ppo --env GridWorldStatic-v0 -n 50000 -f logs/ppo_static_50k --tensorboard-log logs/logs_tensorboard_ppo_gridworld_static_50k/

# 3. Entraîner GridWorld Static – 400k
python -m rl_zoo3.train --algo ppo --env GridWorldStatic-v0 -n 400000 -f logs/ppo_static_400k --tensorboard-log logs/logs_tensorboard_ppo_gridworld_static_400k/

# 4. Entraîner GridWorld Moving – 600k (from scratch)
python -m rl_zoo3.train --algo ppo --env GridWorldMoving-v0 -n 600000 -f logs/ppo_moving_600k --tensorboard-log logs/logs_tensorboard_ppo_gridworld_moving_600k/

# 5. Entraîner GridWorld Moving – run long (par exemple 1.6M)
python -m rl_zoo3.train --algo ppo --env GridWorldMoving-v0 -n 1600000 -f logs/ppo_moving_long2 --tensorboard-log logs/logs_tensorboard_ppo_gridworld_moving_long2/

# 6. Visualiser les politiques apprises et générer les GIF
python test_gridworld_ppo.py --mode static
python test_gridworld_ppo.py --mode moving

# 7. Lancer TensorBoard (exemple)
tensorboard --logdir logs/logs_tensorboard_ppo_gridworld_static_50k
tensorboard --logdir logs/logs_tensorboard_ppo_gridworld_moving_600k

# 8. Entraîner et visualiser CartPole
python cartpole_sb3_view.py --algo ppo --timesteps 20000 --episodes 5
```

---

## 10. Conclusion

Ce projet illustre l'utilisation de **PPO** sur des environnements de complexité croissante :

1. **GridWorldStatic-v0** : tâche simple où l'agent converge rapidement (50k steps suffisent).
2. **GridWorldMoving-v0** : environnement non stationnaire qui nécessite beaucoup plus de données (≈1.6M steps pour une convergence stable).
3. **Transfert raté** : le finetune d'un agent Static vers Moving montre les limites du transfert learning quand la tâche change significativement.
4. **CartPole-v1** : référence classique qui converge très rapidement.

Les **GIF** et **courbes TensorBoard** permettent de visualiser qualitativement et quantitativement les politiques apprises et leur convergence.

---

## Liens utiles

- **Dépôt principal** : [https://github.com/ZIADEA/Reinforcement-Learning-Labs](https://github.com/ZIADEA/Reinforcement-Learning-Labs)
- **Projet Seance5/rl_sb** : [https://github.com/ZIADEA/Reinforcement-Learning-Labs/tree/main/Seance5/rl_sb](https://github.com/ZIADEA/Reinforcement-Learning-Labs/tree/main/Seance5/rl_sb)
- **Stable-Baselines3** : [https://stable-baselines3.readthedocs.io/](https://stable-baselines3.readthedocs.io/)
- **rl-baselines3-zoo** : [https://github.com/DLR-RM/rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
