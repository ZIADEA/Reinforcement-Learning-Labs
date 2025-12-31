<div align="center">

<!-- Bannière animée avec gradient -->
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Reinforcement%20Learning%20%26%20Deep%20RL%20Labs&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32&desc=Portfolio%202025%20-%20Winter%20Semester&descAlignY=51&descAlign=50"/>

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-00A67E?style=for-the-badge)
![License](https://img.shields.io/badge/License-Academic-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge&logo=checkmarx&logoColor=white)

**Collection de devoirs, labs et expériences documentant mon cours en Reinforcement Learning et Deep RL (Hiver 2025) a l'ENSAM de Meknes**

<p align="center">
  <a href="#démarrage-rapide">
    <img src="https://img.shields.io/badge/🚀_Démarrage_Rapide-37a779?style=for-the-badge"/>
  </a>
  <a href="#séances-en-un-coup-dœil">
    <img src="https://img.shields.io/badge/📚_Séances-1e90ff?style=for-the-badge"/>
  </a>
  <a href="#galerie-visuelle">
    <img src="https://img.shields.io/badge/🎬_Démos-ff6b6b?style=for-the-badge"/>
  </a>
  <a href="#ressources-visuelles-et-logs">
    <img src="https://img.shields.io/badge/📊_Résultats-f39c12?style=for-the-badge"/>
  </a>
</p>

</div>

<!-- Ligne de séparation avec effet -->
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

<br/>

## ✨ Points Forts

<div align="center">

<!-- Statistiques animées -->
<table>
<tr>
<td align="center">
<img src="https://img.shields.io/badge/5-Séances-blueviolet?style=for-the-badge&logo=googlescholar&logoColor=white"/>
<br/><sub><b>Séances Complètes</b></sub>
</td>
<td align="center">
<img src="https://img.shields.io/badge/10+-Algorithmes-orange?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<br/><sub><b>Algorithmes RL</b></sub>
</td>
<td align="center">
<img src="https://img.shields.io/badge/50+-Expériences-green?style=for-the-badge&logo=atom&logoColor=white"/>
<br/><sub><b>Expériences Réussies</b></sub>
</td>
<td align="center">
<img src="https://img.shields.io/badge/100+-Visualisations-red?style=for-the-badge&logo=chartdotjs&logoColor=white"/>
<br/><sub><b>Graphiques & GIFs</b></sub>
</td>
</tr>
</table>

</div>

<br/>

<table>
<tr>
<td width="50%">

### 📖 **Parcours d'Apprentissage Structuré**

<img src="https://img.shields.io/badge/✓-Code_Complet-success?style=flat-square"/> Code complet et expériences  
<img src="https://img.shields.io/badge/✓-Analyses_Détaillées-success?style=flat-square"/> Analyses détaillées et figures  
<img src="https://img.shields.io/badge/✓-Documentation_README-success?style=flat-square"/> Documentation README complète  

> Chaque dossier `seance` est une unité autonome avec tout le nécessaire pour reproduire les résultats !

</td>
<td width="50%">

### 🧠 **Techniques Couvertes**

```mermaid
graph TD
    A[Reinforcement Learning] --> B[Classique]
    A --> C[Deep RL]
    B --> D[DP, MC, Q-Learning]
    C --> E[DQN Variants]
    C --> F[Policy Gradients PPO]
    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#ffe1f5
```

</td>
</tr>
</table>

### 🎬 Narration Visuelle

<div align="center">

<kbd>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&duration=2000&pause=1000&color=2E9EF7&center=true&vCenter=true&multiline=true&width=800&height=80&lines=Toutes+les+expériences+incluent+des+visualisations+animées;GIFs+%2B+Tableaux+de+bord+interactifs;Observez+le+comportement+sans+exécuter+le+code+!"/>
</kbd>

</div>

## 🚀 Démarrage Rapide

<details open>
<summary><b>⚙️ Configuration de l'Environnement</b></summary>

```powershell
# Activer l'environnement Python
& C:\Users\DJERI\VSCODE\Programmation\python\environnements\rl_venv\Scripts\Activate.ps1
```
</details>

<details>
<summary><b>🗂️ Naviguer vers les Séances</b></summary>

| Séance | Commande | Objectif |
|---------|---------|----------|
| 📚 Seance 1 | `cd seance1` | Fondamentaux RL (MC, DP, PI, VI, Q-Learning) |
| 🎮 Seance 2/4 | `cd seance2` ou `cd Sceance4/minegym` | Expériences GridWorld & DQN |
| 🚀 Seance 5 | `cd Seance5/rl_sb` | PPO + Stable-Baselines3 |

</details>

<details>
<summary><b>▶️ Lancer les Expériences</b></summary>

```bash
# GridWorld Q-Learning experiments
python -m minegym.experiments.liveQL
python -m minegym.experiments.sensitivity_gammaQL
python -m minegym.experiments.sensitivity_grid_sizeQL

# PPO experiments (from Seance5/rl_sb)
cd Seance5/rl_sb
# See Seance5/readme.md for training scripts
```
</details>

## 📚 Séances en un Coup d'Œil

<div align="center">

<!-- Indicateur de progression -->
<img src="https://progress-bar.dev/100/?title=Progression%20Totale&width=600&color=babaca&suffix=%"/>

</div>

<table>
<tr>
<th width="15%">Séance</th>
<th width="25%">Focus</th>
<th width="20%">Techniques</th>
<th width="40%">Artefacts Clés</th>
</tr>

<tr>
<td align="center">
<br/>
<img src="https://img.shields.io/badge/📖_Seance_1-Fondations-4A90E2?style=for-the-badge"/>
<br/><br/>
<a href="seance1">📂 Voir le dossier</a>
<br/><br/>
</td>
<td>
<b>Algorithmes RL Fondamentaux</b>
<br/><br/>
Apprentissage des bases du RL : programmation dynamique, méthodes Monte Carlo, et apprentissage par différence temporelle.
</td>
<td align="center">

![MC](https://img.shields.io/badge/Monte_Carlo-4169E1?style=flat-square&logo=python)
![DP](https://img.shields.io/badge/Dynamic_Programming-2ECC71?style=flat-square&logo=python)
![QL](https://img.shields.io/badge/Q--Learning-E67E22?style=flat-square&logo=python)
![PI](https://img.shields.io/badge/Policy_Iteration-9B59B6?style=flat-square&logo=python)

<br/>
<img src="https://progress-bar.dev/100/?scale=100&title=Complété&width=120&color=2ecc71"/>

</td>
<td>

• Implémentations agents : MC, PI, VI, Q-Learning  
• Scripts de test et validation  
• Environnements Gym personnalisés  

<details>
<summary>📊 Voir les métriques</summary>
<br/>
<code>✓ 4 algorithmes implémentés</code><br/>
<code>✓ 100% tests réussis</code><br/>
<code>✓ Documentation complète</code>
</details>

</td>
</tr>

<tr>
<td align="center">
<br/>
<img src="https://img.shields.io/badge/🎮_Seance_2-GridWorld-FF6B6B?style=for-the-badge"/>
<br/><br/>
<a href="seance2/minegym">📂 Voir le dossier</a>
<br/><br/>
</td>
<td>
<b>Expériences GridWorld</b>
<br/><br/>
Monde paramétrable avec analyse de sensibilité complète sur γ et la taille de grille.
</td>
<td align="center">

![QL](https://img.shields.io/badge/Q--Learning-E67E22?style=flat-square&logo=python)
![Custom](https://img.shields.io/badge/Custom_Env-8E44AD?style=flat-square&logo=openai)
![Analysis](https://img.shields.io/badge/Sensitivity-16A085?style=flat-square&logo=chartdotjs)

<br/>
<img src="https://progress-bar.dev/100/?scale=100&title=Complété&width=120&color=2ecc71"/>

</td>
<td>

• Monde paramétrable (goals, obstacles)  
• Analyse sensibilité γ et taille grille  
• Q-Learning corrigé (paramètre w)  

<details>
<summary>🎬 Voir les animations</summary>
<br/>
<code>✓ 3 expériences majeures</code><br/>
<code>✓ 20+ graphiques générés</code><br/>
<code>✓ Vidéo training live</code>
</details>

</td>
</tr>

<tr>
<td align="center">
<br/>
<img src="https://img.shields.io/badge/🤖_Seance_4-Deep_RL-EE4C2C?style=for-the-badge"/>
<br/><br/>
<a href="Sceance4/minegym">📂 Voir le dossier</a>
<br/><br/>
</td>
<td>
<b>Deep Q-Networks</b>
<br/><br/>
Comparaison rigoureuse entre approche naïve linéaire et DQN complet avec replay buffer.
</td>
<td align="center">

![DQN](https://img.shields.io/badge/DQN-C0392B?style=flat-square&logo=pytorch)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch)
![Neural](https://img.shields.io/badge/Neural_Nets-3498DB?style=flat-square&logo=tensorflow)

<br/>
<img src="https://progress-bar.dev/100/?scale=100&title=Complété&width=120&color=2ecc71"/>

</td>
<td>

• Comparaison Naïf vs DQN complet  
• Architecture flexible (linéaire/MLP)  
• Protocole expérimental rigoureux  

<details>
<summary>⚙️ Voir les configs</summary>
<br/>
<code>✓ 2 architectures testées</code><br/>
<code>✓ Replay buffer + target net</code><br/>
<code>✓ CSV logs détaillés</code>
</details>

</td>
</tr>

<tr>
<td align="center">
<br/>
<img src="https://img.shields.io/badge/🚀_Seance_5-Policy_Gradient-27AE60?style=for-the-badge"/>
<br/><br/>
<a href="Seance5">📂 Voir le dossier</a>
<br/><br/>
</td>
<td>
<b>Méthodes à Gradient de Politique</b>
<br/><br/>
PPO avec Stable-Baselines3 sur GridWorld statique/mobile et CartPole.
</td>
<td align="center">

![PPO](https://img.shields.io/badge/PPO-27AE60?style=flat-square&logo=openai)
![SB3](https://img.shields.io/badge/Stable--Baselines3-F39C12?style=flat-square&logo=python)
![TB](https://img.shields.io/badge/TensorBoard-FF6F00?style=flat-square&logo=tensorflow)

<br/>
<img src="https://progress-bar.dev/100/?scale=100&title=Complété&width=120&color=2ecc71"/>

</td>
<td>

• GridWorld statique/mobile avec PPO  
• Tentative transfer learning  
• Benchmarks CartPole complets  

<details>
<summary>📈 Voir les runs</summary>
<br/>
<code>✓ 8+ runs d'entraînement</code><br/>
<code>✓ Logs TensorBoard complets</code><br/>
<code>✓ 9 GIFs de démonstration</code>
</details>

</td>
</tr>

<tr>
<td align="center">
<br/>
<img src="https://img.shields.io/badge/👾_Seance_3-Pacman-8E44AD?style=for-the-badge"/>
<br/><br/>
<a href="secance3/reinforcement">📂 Voir le dossier</a>
<br/><br/>
</td>
<td>
<b>Projet Pacman</b>
<br/><br/>
Environnements larges avec autograder complet et agents apprenants sophistiqués.
</td>
<td align="center">

![Games](https://img.shields.io/badge/Game_AI-8E44AD?style=flat-square&logo=atari)
![Grading](https://img.shields.io/badge/Autograder-E74C3C?style=flat-square&logo=checkmarx)

<br/>
<img src="https://progress-bar.dev/100/?scale=100&title=Complété&width=120&color=2ecc71"/>

</td>
<td>

• Environnements larges et complexes  
• Autograder complet  
• Agents apprenants sophistiqués  

<details>
<summary>🎯 Voir les features</summary>
<br/>
<code>✓ Multiple layouts</code><br/>
<code>✓ Ghost agents</code><br/>
<code>✓ Test cases complets</code>
</details>

</td>
</tr>

</table>

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">
</div>

## 🎬 Galerie Visuelle

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=F75C7E&center=true&vCenter=true&width=600&lines=Entraînement+Agents+GridWorld+(PPO);Animations+en+Direct+%F0%9F%8E%AC;Résultats+Spectaculaires+%F0%9F%8C%9F" alt="Typing SVG" />

<br/><br/>

### 🟢 GridWorld Goal Statique

<table>
<tr>
<td align="center">
<a href="Seance5/rl_sb/gridworld_runs/gridworld_static_live.gif">
<img src="Seance5/rl_sb/gridworld_runs/gridworld_static_live.gif" width="300" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);"/>
</a>
<br/>
<img src="https://img.shields.io/badge/100k_steps-Success-2ecc71?style=flat-square"/>
<br/><b>Goal Statique</b>
</td>
<td align="center">
<a href="Seance5/rl_sb/gridworld_runs/gridworld_moving_live.gif">
<img src="Seance5/rl_sb/gridworld_runs/gridworld_moving_live.gif" width="300" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);"/>
</a>
<br/>
<img src="https://img.shields.io/badge/100k_steps-Success-2ecc71?style=flat-square"/>
<br/><b>Goal Mobile</b>
</td>
</tr>
<tr>
<td align="center">
<a href="Seance5/rl_sb/gridworld_runs/gridworld_ppo_static_400k_live.gif">
<img src="Seance5/rl_sb/gridworld_runs/gridworld_ppo_static_400k_live.gif" width="300" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);"/>
</a>
<br/>
<img src="https://img.shields.io/badge/400k_steps-Excellent-27ae60?style=flat-square"/>
<br/><b>Goal Statique (Extended)</b>
</td>
<td align="center">
<a href="Seance5/rl_sb/gridworld_runs/test_cartpole.gif">
<img src="Seance5/rl_sb/gridworld_runs/test_cartpole.gif" width="300" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);"/>
</a>
<br/>
<img src="https://img.shields.io/badge/CartPole-Solved-f39c12?style=flat-square"/>
<br/><b>CartPole-v1 (PPO)</b>
</td>
</tr>
</table>

<br/>

<kbd>💡 <b>Astuce</b> : Cliquez sur les GIFs pour les voir en grand !</kbd>

</div>

<br/>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

## 📊 Ressources Visuelles et Logs

<div align="center">

<table>
<tr>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/96/000000/bar-chart.png" width="80"/>
<br/><br/>
<img src="https://img.shields.io/badge/100+-Graphiques-E74C3C?style=for-the-badge"/>
<br/><br/>
<sub>Heatmaps, dashboards, analyses</sub>
</td>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/96/000000/video.png" width="80"/>
<br/><br/>
<img src="https://img.shields.io/badge/9-GIFs_Animés-3498DB?style=for-the-badge"/>
<br/><br/>
<sub>Visualisations d'agents en action</sub>
</td>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/96/000000/discord-logo.png" width="80"/>
<br/><br/>
<img src="https://img.shields.io/badge/TensorBoard-Logs_Complets-FF6F00?style=for-the-badge"/>
<br/><br/>
<sub>Métriques d'entraînement détaillées</sub>
</td>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/96/000000/save.png" width="80"/>
<br/><br/>
<img src="https://img.shields.io/badge/Checkpoints-Sauvegardés-27AE60?style=for-the-badge"/>
<br/><br/>
<sub>Modèles prêts à l'emploi</sub>
</td>
</tr>
</table>

</div>

<br/>

> **💡 Astuce :** Toutes les expériences incluent des visualisations pré-générées — vous pouvez explorer les résultats sans exécuter le code !

<details>
<summary><b>🗂️ Où trouver les ressources visuelles</b></summary>

| Emplacement | Contenu |
|----------|----------|
| 📁 `Sceance4/minegym/figures` | Heatmaps, tableaux de bord, diagnostics exploration/exploitation (DQN/Q-Learning) |
| 📁 `Seance5/rl_sb/gridworld_runs` | GIFs animés des agents GridWorld et CartPole |
| 📁 `seance2/minegym/figures` | Graphiques d'analyse de sensibilité, courbes de convergence |
| 📁 `Seance5/rl_sb/rl-baselines3-zoo/logs` | Logs TensorBoard et checkpoints des modèles |

</details>

## 🔍 Comment Explorer les Résultats

```mermaid
graph LR
    A[📖 Choisir une Séance] --> B[📚 Lire le README]
    B --> C{Voir les résultats?}
    C -->|Oui| D[🖼️ Parcourir figures/]
    C -->|Non| E[▶️ Lancer expériences]
    D --> F[🎯 Analyser résultats]
    E --> F
```

1. **📖 Commencer** dans le dossier de la séance qui vous intéresse
2. **📚 Lire** son README pour le contexte et les commandes  
3. **🖼️ Inspecter** les figures, GIFs et tableaux de bord pré-générés
4. **▶️ Relancer** les scripts pour générer de nouveaux résultats ou tester de nouveaux paramètres

---

<div align="center">

### 🌟 Merci d'avoir visité ce repository !

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&color=2ECC71&center=true&vCenter=true&width=600&lines=Questions+ou+commentaires+%3F;Ouvrez+une+issue+ou+contactez-moi+!;⭐+Star+ce+repo+si+utile+!" alt="Typing SVG" />

<br/><br/>

<a href="#">
  <img src="https://img.shields.io/badge/⬆️_Retour_en_Haut-2E9EF7?style=for-the-badge"/>
</a>

<br/><br/>

<!-- Footer wave -->
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer"/>

</div>

