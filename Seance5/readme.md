<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=20,23,26&height=140&section=header&text=Seance%205&fontSize=48&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=PPO%20%26%20Stable-Baselines3&descAlignY=55&descAlign=50"/>

<br/>

![PPO](https://img.shields.io/badge/Algorithm-PPO-27AE60?style=for-the-badge&logo=openai)
![SB3](https://img.shields.io/badge/Framework-Stable--Baselines3-F39C12?style=for-the-badge&logo=python)
![Gymnasium](https://img.shields.io/badge/Env-Gymnasium-00A67E?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)

<br/>

**Mini-étude PPO sur GridWorld statique/mobile et CartPole avec analyse de convergence**

</div>

<br/>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

## 🎯 Aperçu

Ce dépôt présente une mini-étude expérimentale autour de PPO appliqué à :

1. **GridWorld statique** (`GridWorldStatic-v0`) - goal fixe
2. **GridWorld mobile** (`GridWorldMoving-v0`) - goal qui se déplace
3. **Transfert** - finetuning du Moving à partir d'un agent Static pré-entraîné
4. **CartPole-v1** - environnement de référence

**Technologies** : Stable-Baselines3, rl-baselines3-zoo, environnement GridWorld custom inspiré de [seance2/minegym](../seance2/minegym)

**Analyse** : TensorBoard (convergence récompenses, longueurs épisodes, pertes) et GIFs de visualisation qualitative

## 🚀 Démarrage Rapide

<details open>
<summary><b>⚙️ 1. Structure du Projet</b></summary>

```
Seance5/rl_sb/
├── gridworld_env/          # Package environnement GridWorld
├── gridworld_runs/         # GIFs et vidéos des agents
├── models/ppo/             # Checkpoints des modèles
└── rl-baselines3-zoo/      # Framework d'entraînement
    └── logs/               # TensorBoard logs
```
</details>

<details>
<summary><b>📦 2. Installation de l'Environnement</b></summary>

```bash
cd Seance5/rl_sb/gridworld_env
pip install -e .
```
</details>

<details>
<summary><b>▶️ 3. Lancer un Entraînement</b></summary>

```bash
cd Seance5/rl_sb/rl-baselines3-zoo
python train.py --algo ppo --env GridWorldStatic-v0 --eval-freq 1000
```
</details>

<details>
<summary><b>📊 4. Visualiser avec TensorBoard</b></summary>

```bash
tensorboard --logdir Seance5/rl_sb/rl-baselines3-zoo/logs
```
</details>

## 🎬 Galerie d'Agents Animés

<div align="center">

### 🟢 GridWorld Goal Statique

<table>
<tr>
<td align="center" width="33%">
<img src="rl_sb/gridworld_runs/gridworld_ppo_static_50k_live.gif" width="240"/>
<br/><br/>
<img src="https://img.shields.io/badge/50k_steps-Training-3498db?style=flat-square"/>
</td>
<td align="center" width="33%">
<img src="rl_sb/gridworld_runs/gridworld_static_live.gif" width="240"/>
<br/><br/>
<img src="https://img.shields.io/badge/100k_steps-Converged-2ecc71?style=flat-square"/>
</td>
<td align="center" width="33%">
<img src="rl_sb/gridworld_runs/gridworld_ppo_static_400k_live.gif" width="240"/>
<br/><br/>
<img src="https://img.shields.io/badge/400k_steps-Optimal-27ae60?style=flat-square"/>
</td>
</tr>
</table>

### 🔵 GridWorld Goal Mobile

<table>
<tr>
<td align="center" width="33%">
<img src="rl_sb/gridworld_runs/gridworld_moving_live.gif" width="240"/>
<br/><br/>
<img src="https://img.shields.io/badge/100k_steps-Training-3498db?style=flat-square"/>
</td>
<td align="center" width="33%">
<img src="rl_sb/gridworld_runs/gridworld_ppo_moving_400k_live.gif" width="240"/>
<br/><br/>
<img src="https://img.shields.io/badge/400k_steps-Converged-2ecc71?style=flat-square"/>
</td>
<td align="center" width="33%">
<img src="rl_sb/gridworld_runs/gridworld_ppo_moving_600k_live.gif" width="240"/>
<br/><br/>
<img src="https://img.shields.io/badge/600k_steps-Optimal-27ae60?style=flat-square"/>
</td>
</tr>
</table>

### 🔄 Transfert Learning & CartPole

<table>
<tr>
<td align="center" width="50%">
<img src="rl_sb/gridworld_runs/gridworld_ppo_moving_finetune_live.gif" width="300"/>
<br/><br/>
<img src="https://img.shields.io/badge/Fine--tuning-Static→Moving-9b59b6?style=flat-square"/>
<br/><sub>Transfert depuis agent statique</sub>
</td>
<td align="center" width="50%">
<img src="rl_sb/gridworld_runs/test_cartpole.gif" width="300"/>
<br/><br/>
<img src="https://img.shields.io/badge/CartPole--v1-Solved-f39c12?style=flat-square"/>
<br/><sub>Benchmark de référence</sub>
</td>
</tr>
</table>

</div>

## 📊 Résultats et Analyses

### Environnements Entraînés

<table>
<tr>
<th>Environnement</th>
<th>Steps</th>
<th>Récompense Moyenne</th>
<th>Statut</th>
</tr>
<tr>
<td>GridWorldStatic-v0</td>
<td>50k / 100k / 400k</td>
<td>~0.95</td>
<td><img src="https://img.shields.io/badge/✓-Complete-success?style=flat-square"/></td>
</tr>
<tr>
<td>GridWorldMoving-v0</td>
<td>100k / 400k / 600k</td>
<td>~0.85</td>
<td><img src="https://img.shields.io/badge/✓-Complete-success?style=flat-square"/></td>
</tr>
<tr>
<td>Fine-tuning (Static→Moving)</td>
<td>Variable</td>
<td>~0.70</td>
<td><img src="https://img.shields.io/badge/⚠-Partiel-orange?style=flat-square"/></td>
</tr>
<tr>
<td>CartPole-v1</td>
<td>Standard</td>
<td>~500</td>
<td><img src="https://img.shields.io/badge/✓-Solved-success?style=flat-square"/></td>
</tr>
</table>

### Observations Clés

- 🎯 **GridWorld Statique** : Convergence rapide, politique optimale claire
- 🔄 **GridWorld Mobile** : Apprentissage plus long, adaptation nécessaire
- ⚠️ **Transfert Learning** : Performance limitée, nécessite réentraînement significatif
- ✅ **CartPole** : Validation du pipeline d'entraînement PPO

## 📁 Ressources Disponibles

<table>
<tr>
<td width="33%" align="center">
<br/>
🎬 <b>Animations</b>
<br/><br/>
9 GIFs dans<br/><code>gridworld_runs/</code>
<br/><br/>
</td>
<td width="33%" align="center">
<br/>
📈 <b>TensorBoard Logs</b>
<br/><br/>
Logs complets dans<br/><code>rl-baselines3-zoo/logs/</code>
<br/><br/>
</td>
<td width="33%" align="center">
<br/>
💾 <b>Checkpoints</b>
<br/><br/>
Modèles dans<br/><code>models/ppo/</code>
<br/><br/>
</td>
</tr>
</table>

## 🔧 Configuration PPO Utilisée

```python
{
    "policy": "MlpPolicy",
    "n_steps": 2048,
    "batch_size": 64,
    "gae_lambda": 0.95,
    "gamma": 0.99,
    "n_epochs": 10,
    "ent_coef": 0.0,
    "learning_rate": 3e-4,
    "clip_range": 0.2
}
```

## 🔍 Comment Explorer

1. **Consulter les GIFs** dans `gridworld_runs/` pour voir les agents en action
2. **Analyser TensorBoard** : `tensorboard --logdir rl-baselines3-zoo/logs`
3. **Tester les modèles** : utiliser `enjoy.py` de rl-baselines3-zoo
4. **Réentraîner** : modifier hyperparamètres et relancer `train.py`

---

<div align="center">

<br/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&color=27AE60&center=true&vCenter=true&width=500&lines=PPO+sur+GridWorld+%E2%9C%85;CartPole+Solved+%E2%9C%85;8%2B+Training+Runs" alt="Typing SVG" />

<br/><br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=20,23,26&height=100&section=footer"/>

</div>
