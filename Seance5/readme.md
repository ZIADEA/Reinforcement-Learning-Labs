# RL GridWorld + CartPole – PPO, Fine-Tuning et Analyse TensorBoard

Ce dépôt contient une petite étude expérimentale de PPO appliqué à :

1. Un environnement **GridWorld statique** (goal fixe)
2. Un environnement **GridWorld avec goal mobile**
3. Un cas de **mauvais transfert** (finetuning Moving à partir de Static)
4. Un environnement de référence **CartPole-v1**

L’implémentation est basée sur **Stable-Baselines3** et **rl-baselines3-zoo**, avec un environnement GridWorld custom et une analyse de convergence via **TensorBoard**.

---

## 1. Fichiers importants du dépôt

### 1.1. Environnement GridWorld custom

- `gridworld_env/gridworld_env/grid_core.py`  
  Logique interne de la grille (indexation, déplacement de l’agent, gestion des goals, récompenses, etc.).

- `gridworld_env/gridworld_env/grid_sb3_env.py`  
  Wrapper Gymnasium / SB3 :
  - enregistre les environnements `GridWorldStatic-v0` et `GridWorldMoving-v0`
  - fournit `reset()` / `step()` compatibles rl-baselines3-zoo
  - convertit l’état en observation entière.

- `gridworld_env/gridworld_env/__init__.py`  
  Point d’entrée du package `gridworld_env` utilisé par les scripts d’entraînement/test.

### 1.2. Scripts d’entraînement / visualisation

Dans le dossier du zoo : `rl-baselines3-zoo/`

- `rl-baselines3-zoo/train.py`  
  Script d’entraînement générique du zoo (appelé via `python -m rl_zoo3.train`).

- `rl-baselines3-zoo/test_gridworld_ppo.py`  
  Script maison pour :
  - charger un modèle PPO sur `GridWorldStatic-v0` ou `GridWorldMoving-v0`
  - afficher l’agent en live avec Matplotlib (fond blanc, agent rouge, goal vert)
  - enregistrer des GIF dans `gridworld_runs/`
  - tracer dans le terminal : action, reward, position de l’agent, position du goal.

- `rl-baselines3-zoo/cartpole_sb3_view.py`  
  Script maison pour CartPole :
  - entraîner un PPO simple sur `CartPole-v1`
  - sauvegarder le modèle
  - rejouer quelques épisodes en mode `render="human"`.

### 1.3. Hyperparamètres PPO

- `rl-baselines3-zoo/hyperparams/ppo.yml`  

  Fichier modifié pour ajouter des entrées spécifiques :

  ```yaml
  GridWorldStatic-v0:
    n_steps: 256
    batch_size: 64
    learning_rate: 0.0003
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.2
    ent_coef: 0.0

  GridWorldMoving-v0:
    n_steps: 256
    batch_size: 64
    learning_rate: 0.0003
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.2
    ent_coef: 0.0
  ```

  Ces configurations sont automatiquement utilisées par `rl_zoo3.train` lorsqu’on spécifie `--env GridWorldStatic-v0` ou `--env GridWorldMoving-v0`.

### 1.4. Modèles entraînés (checkpoints PPO)

Les modèles PPO principaux sont stockés ici :

- Static :  
  `rl-baselines3-zoo/logs/ppo/GridWorldStatic-v0_1/`  
  - `best_model.zip`  
  - `GridWorldStatic-v0.zip`  
  - `0.monitor.csv`  
  - `config.yml`, `args.yml`, `command.txt`

- Moving entraîné **directement depuis PPO vierge** :  
  `rl-baselines3-zoo/logs/ppo/GridWorldMoving-v0_1/`

- Moving **finetuné à partir du modèle Static** (cas “raté”) :  
  `rl-baselines3-zoo/logs/ppo/GridWorldMoving-v0_1_fintune_sur_gridword_static/`

- CartPole :  
  `rl-baselines3-zoo/logs/ppo/CartPole-v1_1/`  
  Et copie du modèle dans :  
  `models/ppo/CartPole-v1/CartPole-v1.zip`

### 1.5. Logs TensorBoard

Logs bruts au format `events.out.tfevents...` :

- Static :  
  `rl-baselines3-zoo/logs/logs_tensorboard_ppo_gridworld_static/GridWorldStatic-v0/PPO_1/`

- Moving finetuné sur Static :  
  `rl-baselines3-zoo/logs/logs_tensorboard_ppo_gridworld_moving_fintune_sur_gridword_static/GridWorldMoving-v0/PPO_1/`

- Moving entraîné directement :  
  `rl-baselines3-zoo/logs/logs_tensorboard_ppo_gridworld_finetune_moving/GridWorldMoving-v0/PPO_1/`

- CartPole :  
  `rl-baselines3-zoo/logs/test_cartpole_tb/PPO_1/`

Un zip global est également présent :  
`rl-baselines3-zoo/logs/tensorbord.zip`  
(il contient les logs convertis en CSV pour analyse hors TensorBoard).

### 1.6. GIF des politiques apprises

Tous les GIF sont regroupés dans :

- `gridworld_runs/`

Fichiers principaux :

1. `gridworld_static_live.gif`  
   → Agent PPO sur `GridWorldStatic-v0` (goal fixe)

2. `gridworld_moving_live_fintune_sur_gridword_static.gif`  
   → Moving finetuné à partir du modèle Static (cas de transfert raté)

3. `gridworld_moving_live.gif`  
   → Moving entraîné directement sur PPO vierge (cas final correct)

4. `test_cartpole.gif`  
   → PPO sur CartPole-v1 (référence)

---

## 2. Description des environnements GridWorld

### 2.1. GridWorldStatic-v0

- Grille 4×4
- Un seul goal fixe (récompense positive lorsqu’il est atteint)
- Récompenses typiques :
  - −1 par step
  - +30 lorsque le goal est atteint
- Utilisé comme tâche de base pour initialiser un agent PPO.

### 2.2. GridWorldMoving-v0

- Même grille 4×4
- Le goal change de position à chaque step (mouvement aléatoire)
- Tâche plus difficile : l’agent doit “poursuivre” une cible mobile.

---

## 3. Commandes d’entraînement

Toutes les commandes suivantes se lancent depuis :

```bash
cd rl-baselines3-zoo
```

### 3.1. PPO sur GridWorldStatic-v0

Entraînement depuis zéro :

```bash
python -m rl_zoo3.train --algo ppo --env GridWorldStatic-v0     -n 100000     --tensorboard-log logs/ppo/
```

Les résultats sont sauvegardés dans :  
`logs/ppo/GridWorldStatic-v0_1/`

### 3.2. PPO sur GridWorldMoving-v0 – finetune à partir du modèle Static (cas raté)

```bash
python -m rl_zoo3.train --algo ppo --env GridWorldMoving-v0     -i logs/ppo/GridWorldStatic-v0_1/GridWorldStatic-v0.zip     -n 50000     --tensorboard-log logs/ppo/
```

Résultats :  
`logs/ppo/GridWorldMoving-v0_1_fintune_sur_gridword_static/`

Remarque :  
Cette expérience montre un **exemple de transfert qui se passe mal**.  
L’agent a déjà appris sur Static à “aller au coin 4×4” et garde ce biais, sans réussir à suivre un goal qui bouge.

### 3.3. PPO sur GridWorldMoving-v0 – entraînement direct (cas correct)

Entraînement depuis des poids PPO vierges :

```bash
python -m rl_zoo3.train --algo ppo --env GridWorldMoving-v0     -n 100000     --tensorboard-log logs/ppo/
```

Résultats :  
`logs/ppo/GridWorldMoving-v0_1/`

Ici l’agent apprend réellement la stratégie “poursuivre un goal mobile” sans être prisonnier de la politique apprise sur Static.

### 3.4. PPO sur CartPole-v1 (référence)

Script dédié :

```bash
cd rl-baselines3-zoo
python cartpole_sb3_view.py --algo ppo --timesteps 20000 --episodes 5
```

- Sauvegarde des modèles et logs PPO standard dans  
  `logs/ppo/CartPole-v1_1/`
- Copie du modèle dans  
  `models/ppo/CartPole-v1/CartPole-v1.zip`
- Génération du GIF : `gridworld_runs/test_cartpole.gif`

---

## 4. Visualisation des politiques apprises (test + GIF)

Script de test GridWorld :  
`rl-baselines3-zoo/test_gridworld_ppo.py`

### 4.1. Tester Static

```bash
cd rl-baselines3-zoo
python test_gridworld_ppo.py --mode static
```

Utilise par défaut :

- Modèle : `logs/ppo/GridWorldStatic-v0_1/best_model.zip`
- Environnement : `GridWorldStatic-v0`
- GIF généré : `gridworld_runs/gridworld_static_live.gif`

### 4.2. Tester Moving finetuné sur Static

Modifier le chemin du modèle dans `test_gridworld_ppo.py` si besoin, puis :

```bash
python test_gridworld_ppo.py --mode moving
```

et utiliser :

- Modèle : `logs/ppo/GridWorldMoving-v0_1_fintune_sur_gridword_static/best_model.zip`
- GIF : `gridworld_runs/gridworld_moving_live_fintune_sur_gridword_static.gif`

On observe que l’agent reste souvent bloqué sur une cellule (type 4×4) :  
il “croit” que le goal est toujours à cet endroit, même s’il bouge.

### 4.3. Tester Moving entraîné directement

Même commande, mais modèle :

- `logs/ppo/GridWorldMoving-v0_1/best_model.zip`
- GIF : `gridworld_runs/gridworld_moving_live.gif`

Cette fois l’agent suit correctement la position du goal mobile dans la grille.

---

## 5. Interprétation de la convergence (TensorBoard)

Les fichiers TensorBoard se trouvent dans :

- Static : `logs/logs_tensorboard_ppo_gridworld_static/...`
- Moving finetune sur Static : `logs/logs_tensorboard_ppo_gridworld_moving_fintune_sur_gridword_static/...`
- Moving from scratch : `logs/logs_tensorboard_ppo_gridworld_finetune_moving/...`
- CartPole : `logs/test_cartpole_tb/...`

Les scalaires principaux analysés sont :

- `rollout/ep_rew_mean` (récompense moyenne)
- `rollout/ep_len_mean` (longueur d’épisode moyenne)
- `train/value_loss`
- `train/policy_gradient_loss`

### 5.1. GridWorldStatic-v0

- La récompense moyenne augmente rapidement puis se stabilise :
  l’agent atteint le goal en quelques dizaines de steps.
- La valeur moyenne des épisodes devient stable : bonne estimation de la fonction de valeur.
- Le `value_loss` diminue puis oscille sur une plage basse.

Conclusion :  
**Convergence rapide et propre** sur la tâche simple (goal fixe).

### 5.2. GridWorldMoving – finetune sur Static

- La récompense moyenne reste relativement faible ; les épisodes sont longs.
- On observe un `value_loss` qui fluctue sans trouver de vrai plateau.
- Le `policy_gradient_loss` est faible : l’agent ne modifie presque plus sa politique.

Interprétation :  
L’agent est **piégé par la politique apprise sur Static**.  
Pour lui, la bonne stratégie reste “aller au coin où se trouvait le goal fixe”, même si le goal bouge.  
Le finetuning part d’un optimum local qui n’est plus pertinent → convergence très mauvaise.

### 5.3. GridWorldMoving – entraînement direct

- La récompense moyenne augmente progressivement puis se stabilise à un niveau clairement supérieur au cas finetune.
- La longueur moyenne des épisodes diminue : l’agent trouve le goal plus vite.
- Les courbes de `value_loss` et `policy_gradient_loss` deviennent stables après une phase transitoire.

Interprétation :  
En partant d’un PPO **vierge**, l’agent apprend vraiment la stratégie de poursuite d’un goal mobile.  
On obtient une **convergence correcte, sans biais hérité de Static**.

### 5.4. CartPole-v1

- Convergence très rapide vers des récompenses proches du max.
- Épisodes très longs (souvent terminés par le time-limit et non par chute du pendule).
- Courbes stables, comportement attendu pour un PPO standard.

---

## 6. Comparaison qualitative (GIF)

Pour un aperçu rapide, comparer les GIF dans `gridworld_runs/` :

| Expérience                            | Fichier GIF                                            | Comportement observé |
|--------------------------------------|--------------------------------------------------------|----------------------|
| GridWorld Static                     | `gridworld_static_live.gif`                            | L’agent va directement vers le goal fixe. |
| GridWorld Moving – finetune Static   | `gridworld_moving_live_fintune_sur_gridword_static.gif`| L’agent reste bloqué sur une case, ignore le goal qui bouge. |
| GridWorld Moving – from scratch      | `gridworld_moving_live.gif`                            | L’agent suit correctement la position du goal mobile. |
| CartPole-v1                          | `test_cartpole.gif`                                    | Le pendule reste globalement équilibré, le chariot oscille légèrement. |

---

## 7. Reproduction rapide

Résumé minimal pour rejouer les expériences :

```bash
# 1. Se placer dans le dossier du zoo
cd rl-baselines3-zoo

# 2. Entraîner GridWorld static
python -m rl_zoo3.train --algo ppo --env GridWorldStatic-v0 -n 100000 --tensorboard-log logs/ppo/

# 3. Entraîner GridWorld moving (version "correcte")
python -m rl_zoo3.train --algo ppo --env GridWorldMoving-v0 -n 100000 --tensorboard-log logs/ppo/

# 4. Visualiser et générer les GIF
python test_gridworld_ppo.py --mode both

# 5. Entraîner et visualiser CartPole
python cartpole_sb3_view.py --algo ppo --timesteps 20000 --episodes 5
```

---

Ce README se concentre uniquement sur les fichiers et dossiers utiles à la compréhension des expériences menées (GridWorld static, GridWorld moving, finetuning, CartPole, TensorBoard et GIF).  
