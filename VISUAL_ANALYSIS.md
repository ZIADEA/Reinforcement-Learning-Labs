<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=140&section=header&text=Analyse%20Visuelle%20Complète&fontSize=42&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=Catalogue%20des%2063%20Fichiers%20Visuels&descAlignY=55&descAlign=50"/>

<br/>

![Images](https://img.shields.io/badge/Images_PNG-45-3498db?style=for-the-badge&logo=files)
![GIFs](https://img.shields.io/badge/GIFs-8-2ecc71?style=for-the-badge&logo=giphy)
![CSV](https://img.shields.io/badge/CSV_Logs-~10-f39c12?style=for-the-badge&logo=databricks)

**Document récapitulatif de tous les assets visuels avec analyses et justifications**

</div>

<br/>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

## 📋 Table des Matières

1. [Séance 2 - GridWorld Q-Learning](#seance-2---gridworld-q-learning-27-png--1-csv)
2. [Séance 4 - DQN Flexible](#seance-4---dqn-flexible-10-png--1-csv)
3. [Séance 5 - PPO Stable-Baselines3](#seance-5---ppo-stable-baselines3-8-gifs--8-png--8-csv)
4. [Récapitulatif Global](#recapitulatif-global)

---

## Séance 2 - GridWorld Q-Learning (27 PNG + 1 CSV)

### 📂 `liveQLgoalsfixed/` (7 fichiers)

| # | Fichier | Type | Dimensions | Description | Utilité |
|---|---------|------|-----------|-------------|----------|
| 1 | `V_star_heatmap_annotated.png` | PNG | ~800×600 | **Heatmap de V\*** avec annotations (goal, obstacles, valeurs numériques) | Visualiser la fonction de valeur optimale : zones chaudes = proches du goal, froides = loin. Valide que l'agent a appris la structure de récompense. |
| 2 | `pi_star_grid.png` | PNG | ~800×600 | **Politique optimale π\*** (grille de flèches ←→↑↓) | Vérifier que chaque flèche pointe vers le goal. Détecte les incohérences (flèches vers obstacles). |
| 3 | `policy_value.png` | PNG | ~1000×600 | **Combinaison π\* + V\*** (overlay flèches + heatmap) | Vue unifiée : valide que arg max Q(s,a) extrait bien l'action qui maximise V*. |
| 4 | `visits.png` | PNG | ~800×600 | **Heatmap des visites d'états** (nb de fois où agent passe par chaque case) | Identifier biais d'exploration : zones sur-explorées vs négligées. Utile pour détecter si ε-greedy couvre uniformément l'espace. |
| 5 | `dominant_actions.png` | PNG | ~800×600 | **Action dominante empirique** (action la + fréquente par état) | Complément à π* : révèle le comportement empirique vs théorique (ε-greedy résiduel biaise certaines actions). |
| 6 | `summary_dashboard.png` | PNG | ~1600×1200 | **Dashboard 6-en-1** : reward distribution, convergence, episode length, action distribution, explore/exploit, reward/step | **Vue synthétique complète** : diagnostic rapide de 6 métriques clés. Identifie problèmes (plateau, biais action, exploration insuffisante). |
| 7 | `live_explore_exploit_empirical.png` | PNG | ~900×600 | **Proportion actions greedy** (empirique vs théorique ε) | Valide que l'agent respecte le schedule d'ε (courbe empirique ≈ théorique). Détecte si stochastique de l'env force plus d'exploration. |

**📝 Génération** : Script `liveQL.py` — entraîne Q-Learning avec dashboard animé, génère MP4 + 7 PNGs diagnostiques.

---

### 📂 `sensitivity_gamma/` (6 fichiers)

| # | Fichier | Type | Description | Utilité |
|---|---------|------|-------------|----------|
| 8 | `sensitivity_gamma_convergence_ci.png` | PNG | **Courbes convergence** pour γ ∈ [0.0, 0.99] (11 valeurs), MA50, avec intervalles de confiance (5 seeds) | **Comparer vitesse et qualité finale** selon γ : γ faibles → rapide mais sous-optimal ; γ élevés → lent mais optimal. Guide choix hyperparamètre. |
| 9 | `sensitivity_gamma_time_to_threshold.png` | PNG | **Barplot temps pour atteindre seuil** (MA50 ≥ -10) par γ | Quantifier "quelle γ converge le plus vite ?". Révèle que γ=0.5-0.7 optimal pour vitesse. |
| 10 | `sensitivity_gamma_final.png` | PNG | **Performance finale** (moyenne 200 derniers épisodes) avec barres d'erreur | Confirme que γ=0.99 atteint ~+40 (optimal) vs γ=0.1 plafonne à ~-5. |
| 11 | `sensitivity_gamma_episode_length.png` | PNG | **Longueur épisodes** (MA50) vs épisodes, pour 1 seed illustratif | Montre que γ élevés → trajectoires + longues initialement (exploration profonde) puis stabilisation. |
| 12 | `sensitivity_gamma_explore_exploit.png` | PNG | **Profil ε théorique** (planification epsilon_schedule) | Baseline : tous suivent même décroissance d'ε (différences viennent du γ, pas de l'exploration). |
| 13 | `sensitivity_gamma_prop_greedy_subplots.png` | PNG | **11 subplots** (1 par γ) : proportion actions greedy empirique vs épisodes | Révèle que γ=0.0 reste <80% greedy (politique instable), autres convergent >90%. |

**📝 Génération** : Script `sensitivity_gammaQL.py` — sweep γ avec 5 seeds, 2000 épisodes, grille 10×10.

---

### 📂 `sensitivity_grid_size/` (5 fichiers)

| # | Fichier | Type | Description | Utilité |
|---|---------|------|-------------|----------|
| 14 | `sensitivity_grid_convergence.png` | PNG | **Courbes convergence** pour tailles 4×4, 6×6, 8×8, 10×10 | Quantifier scaling laws : 4×4 converge en 100 ep, 10×10 en 1000 ep → scaling ~quadratique. |
| 15 | `sensitivity_grid_final.png` | PNG | **Performance finale** par taille | Confirme reward final identique (~+40) car envs normalisés (même goal_reward ratio). |
| 16 | `sensitivity_grid_episode_length.png` | PNG | **Longueur épisodes** par taille | Croissance linéaire : 4×4 ~8 steps, 10×10 ~25 steps (distance Manhattan augmente). |
| 17 | `sensitivity_grid_explore_exploit.png` | PNG | **Profil ε théorique** commun | Baseline : même schedule pour toutes tailles (différences = complexité state space). |
| 18 | `sensitivity_grid_prop_greedy_subplots.png` | PNG | **7 subplots** (1 par taille) : proportion greedy | 4×4 atteint 95% dès ep 200, 10×10 vers ep 800 → exploration + longue sur grands espaces. |

**📝 Génération** : Script `sensitivity_grid_sizeQL.py` — teste 7 tailles avec obstacles diagonaux.

---

### 📂 `exp_td0_linear/` (3 PNG + 1 CSV)

| # | Fichier | Type | Description | Utilité |
|---|---------|------|-------------|----------|
| 19 | `td0_Vpi_heatmap.png` | PNG | **Heatmap Vπ** (fonction valeur sous politique fixe) | Baseline prédiction : montre que politique aléatoire donne valeurs négatives partout (pas de gradient vers goal). |
| 20 | `td0_pi_followed.png` | PNG | **Politique suivie** (fixe, imposée) | Visualiser la politique testée (flèches désordonnées car aléatoire). |
| 21 | `td0_convergence.png` | PNG | **Convergence V(s₀)** vs épisodes | Stabilisation rapide ~200 ep de V(s₀) autour de -20 (politique sous-optimale). |
| 22 | `td0_pi.csv` | CSV | Politique testée (format : state, action) | Reproductibilité : permet re-tester exacte même politique. |

**📝 Génération** : Script `exp_td0_linear.py` — évalue politique fixe avec TD(0).

---

### 📂 `exp_sarsa_linear/` (3 fichiers)

| # | Fichier | Type | Description | Utilité |
|---|---------|------|-------------|----------|
| 23 | `sarsa_V_heatmap.png` | PNG | **V\* (SARSA)** | Comparaison avec TD(0) : valeurs positives près goal → gradient clair (control vs prédiction). |
| 24 | `sarsa_pi_grid.png` | PNG | **π\* (SARSA)** | Flèches convergent vers goal mais avec quelques détours (on-policy prudent). |
| 25 | `sarsa_returns.png` | PNG | **Retours par épisode** | Croissance progressive jusqu'à +30-40 (convergence on-policy). |

**📝 Génération** : Script `exp_sarsa_linear.py` — SARSA control avec approximation linéaire.

---

### 📂 `exp_dqn/` (3 fichiers)

| # | Fichier | Type | Description | Utilité |
|---|---------|------|-------------|----------|
| 26 | `dqn_V_heatmap.png` | PNG | **V\* (DQN)** | Valeurs positives optimales (+45-50) → meilleur que SARSA (off-policy avantage). |
| 27 | `dqn_pi_grid.png` | PNG | **π\* (DQN)** | Flèches parfaitement alignées vers goal (politique optimale). |
| 28 | `dqn_returns.png` | PNG | **Retours par épisode** | Croissance plus lente mais atteint +45-50 (meilleur final). |

**📝 Génération** : Script `exp_dqn.py` — DQN avec replay buffer et target network.

---

## Séance 4 - DQN Flexible (10 PNG + 1 CSV)

### 📂 `flex_naive_lin/` (11 fichiers)

| # | Fichier | Type | Dimensions | Description | Utilité |
|---|---------|------|-----------|-------------|----------|
| 29 | `V_star_heatmap.png` | PNG | ~800×600 | **V\* apprise** (mode naive linéaire) | Valide que naive apprend gradient malgré absence replay buffer (env 6×6 assez simple). |
| 30 | `pi_star_grid.png` | PNG | ~800×600 | **π\* (naive)** | Politique cohérente → convergence OK sur petit espace d'états. |
| 31 | `policy_value.png` | PNG | ~1000×600 | **π\* + V\* overlay** | Validation que arg max Q suit gradient V*. |
| 32 | `naive_loss_mean_per_episode.png` | PNG | ~900×600 | **TD-loss MSE** par épisode (MA50) | Décroissance de ~30 → ~10 après 800 ep → optimiseur converge (résidu = ε-greedy stochastique). |
| 33 | `naive_steps_per_episode.png` | PNG | ~900×600 | **Steps/épisode** (MA50) | Décroissance de ~40 → ~12-15 steps (optimal Manhattan). Proxy de performance. |
| 34 | `naive_epsilon_theta_over_episodes.png` | PNG | ~900×600 | **Dual-axis** : ε (bleu) + \\|\\|θ\\|\\| (orange) | Valide décroissance ε et stabilisation poids (~8-10) → pas de divergence catastrophique. |
| 35 | `naive_loss_vs_return.png` | PNG | ~800×800 | **Scatter plot** loss vs return | Corrélation négative : return élevé → loss faible. Diagnostique qualité approximateur Q. |
| 36 | `visits.png` | PNG | ~800×600 | **Heatmap visites** | Distribution uniforme (goals aléatoires forcent exploration complète). |
| 37 | `dominant_actions.png` | PNG | ~800×600 | **Actions dominantes** | Cohérence avec π* (empirique ≈ théorique). |
| 38 | `summary_dashboard.png` | PNG | ~1600×1200 | **Dashboard 6-en-1** | Vue synthétique : reward, convergence, actions, explore/exploit. |
| 39 | `naive_logs.csv` | CSV | - | **Logs par épisode** : episode, steps, return, loss, epsilon, theta_norm | Export pour analyses Pandas/Excel. Reproductibilité exacte. |

**📝 Génération** : Script `exp_flexible.py --mode naive` — DQN linéaire sans replay/target.

---

## Séance 5 - PPO Stable-Baselines3 (8 GIFs + 8 PNG + ~8 CSV)

### 📂 `gridworld_runs/` (8 GIFs)

| # | Fichier | Type | Durée | FPS | Description | Utilité |
|---|---------|------|-------|-----|-------------|----------|
| 40 | `gridworld_ppo_static_50k_live.gif` | GIF | ~5s | 12 | Agent Static 50k steps (en apprentissage) | Hésitations visibles, quelques détours → pas encore optimal. |
| 41 | `gridworld_static_live.gif` | GIF | ~4s | 12 | Agent Static 100k steps (convergé) | Trajectoires directes vers goal → convergence visible. |
| 42 | `gridworld_ppo_static_400k_live.gif` | GIF | ~3s | 12 | Agent Static 400k steps (expert) | Trajectoires parfaites, 12-15 steps optimal. |
| 43 | `gridworld_moving_live.gif` | GIF | ~6s | 12 | Agent Moving 100k steps | Suit goal avec ~2-3 steps de retard. |
| 44 | `gridworld_ppo_moving_400k_live.gif` | GIF | ~5s | 12 | Agent Moving 400k steps | Réactivité améliorée, suit goal rapidement. |
| 45 | `gridworld_ppo_moving_600k_live.gif` | GIF | ~4s | 12 | Agent Moving 600k steps | Réactivité quasi-instantanée, adaptation optimale. |
| 46 | `gridworld_ppo_moving_finetune_live.gif` | GIF | ~5s | 12 | Agent Fine-tune (Static→Moving) | Performances intermédiaires ~70% du natif. |
| 47 | `test_cartpole.gif` | GIF | ~8s | 12 | CartPole-v1 solved | Équilibre stable >500 steps (validation pipeline PPO). |

**📝 Génération** : Scripts `enjoy.py` de rl-baselines3-zoo avec modèles entraînés.

**💡 Utilité Globale** : Validation qualitative que métriques (reward, steps) reflètent comportement observé.

---

### 📂 `images/` (8 PNG TensorBoard)

| # | Fichier | Type | Dimensions | Description | Utilité |
|---|---------|------|-----------|-------------|----------|
| 48 | `static_50k_400k_ep_rew_mean.png` | PNG | ~1200×800 | **Comparaison reward** Static 50k vs 400k | Montre que 50k atteint ~0.90, 400k stabilise à ~0.95 → 50k suffisant pour résoudre. |
| 49 | `static_50k_400k_ep_len_mean.png` | PNG | ~1200×800 | **Comparaison episode length** Static | Décroissance de ~25 → ~12-15 steps (optimal Manhattan). |
| 50 | `gridworld_static_50k_ep_rew_mean.png` | PNG | ~900×600 | **Snapshot reward 50k** | Vue isolée 50k steps. |
| 51 | `gridworld_static_400k_ep_rew_mean.png` | PNG | ~900×600 | **Snapshot reward 400k** | Vue isolée 400k steps. |
| 52 | `moving_400k_600k_1600k_ep_rew_mean.png` | PNG | ~1400×800 | **Comparaison reward** Moving 3 runs | 400k ~0.75, 600k ~0.85, 1600k ~0.90 → convergence 10× plus lente que static. |
| 53 | `moving_400k_600k_1600k_ep_len_mean.png` | PNG | ~1400×800 | **Comparaison episode length** Moving | Reste ~15-17 steps (légèrement > static car réactivité au goal mobile). |
| 54 | `gridworld_moving_400k_ep_rew_mean.png` | PNG | ~900×600 | **Snapshot reward Moving 400k** | Vue isolée 400k. |
| 55 | `gridworld_moving_600k_ep_rew_mean.png` | PNG | ~900×600 | **Snapshot reward Moving 600k** | Vue isolée 600k. |

**📝 Source** : TensorBoard logs exportés en PNG (clics manuels ou `tensorboard --logdir logs`).

**💡 Utilité** : Quantifier différence Static vs Moving (scaling factor ~10×), valider que convergence suit trajectoires GIFs.

---

### 📂 `logs/ppo/.../` (~8 CSV monitor.csv)

**Structure type** :
```csv
# {"t_start": 1234567890.0, "env_id": "GridWorldStatic-v0"}
r,l,t
-5.0,23,0.12
8.5,17,0.25
45.0,12,0.38
```

**Colonnes** : `r` (reward cumulé), `l` (longueur épisode), `t` (timestamp)

**💡 Utilité** : Import Pandas pour analyses custom (variance, quantiles, t-tests multi-runs).

---

## Récapitulatif Global

### 📊 Distribution par Séance

<div align="center">

| Séance | PNG | GIF | CSV | Total | Ratio |
|--------|-----|-----|-----|-------|-------|
| **Seance 2** | 27 | 0 | 1 | 28 | 44% |
| **Seance 4** | 10 | 0 | 1 | 11 | 17% |
| **Seance 5** | 8 | 8 | ~8 | ~24 | 38% |
| **TOTAL** | **45** | **8** | **~10** | **~63** | **100%** |

</div>

### 🎯 Utilités Principales par Catégorie

<table>
<tr>
<th width="25%">Catégorie</th>
<th width="15%">Nb Files</th>
<th width="60%">Justification Utilité</th>
</tr>
<tr>
<td>🌡️ <b>Heatmaps V*</b></td>
<td>8</td>
<td>Valider fonction valeur optimale apprise : gradient clair vers goal, détecte sous-optimalité, compare algos (TD/SARSA/DQN/Naive).</td>
</tr>
<tr>
<td>🧭 <b>Politiques π*</b></td>
<td>8</td>
<td>Vérifier cohérence flèches → goal, identifier biais (flèches vers obstacles), comparer on-policy vs off-policy.</td>
</tr>
<tr>
<td>📈 <b>Courbes Convergence</b></td>
<td>15</td>
<td>Quantifier vitesse convergence, comparer hyperparamètres (γ, grid size), détecter plateaux prématurés.</td>
</tr>
<tr>
<td>📏 <b>Episode Length</b></td>
<td>6</td>
<td>Proxy performance : décroissance → politique plus directe. Valide que agent atteint optimal Manhattan.</td>
</tr>
<tr>
<td>🎢 <b>Loss/TD-error</b></td>
<td>3</td>
<td>Diagnostic optimiseur : décroissance → convergence, pics → instabilité, résidu → stochasticité env.</td>
</tr>
<tr>
<td>⚖️ <b>Explore/Exploit</b></td>
<td>4</td>
<td>Valider schedule ε, détecter exploration insuffisante (plateau prématuré) ou excessive (convergence lente).</td>
</tr>
<tr>
<td>🗺️ <b>Visits Heatmaps</b></td>
<td>3</td>
<td>Identifier zones sous-explorées (trous coverage) ou sur-visitées (biais ε-greedy).</td>
</tr>
<tr>
<td>📊 <b>Dashboards</b></td>
<td>3</td>
<td>Vue synthétique 6-en-1 : diagnostic rapide multi-métrique, gain temps vs lectures multiples.</td>
</tr>
<tr>
<td>🔗 <b>Scatter Plots</b></td>
<td>1</td>
<td>Corrélation loss vs return : diagnostique qualité approximateur Q (surestimation/sous-estimation biais).</td>
</tr>
<tr>
<td>🎬 <b>GIFs Animés</b></td>
<td>8</td>
<td>Validation qualitative : comportement observé ≈ métriques. Détecte problèmes invisibles dans curves (oscillations, boucles).</td>
</tr>
<tr>
<td>📂 <b>CSVs Logs</b></td>
<td>~10</td>
<td>Reproductibilité, analyses externes (Pandas, Excel), statistiques avancées (variance inter-runs, t-tests).</td>
</tr>
</table>

### 🔍 Méta-Analyse : Patterns Transversaux

**1. Convergence Multi-Algo** :
- **TD(0)** : Rapide (~200 ep) mais sous-optimal (prédiction ≠ control)
- **SARSA** : Moyenne (~500 ep), on-policy prudent
- **DQN** : Lente (~1000 ep) mais meilleur final (+45-50 vs +30-40)
- **Naive DQN** : Similaire à DQN sur petit env (6×6), divergerait sur grand env
- **PPO** : Très rapide sur Static (50k), 10× plus lent sur Moving (600k)

**2. Sensibilité Hyperparamètres** :
- **γ** : 0.5-0.7 optimal vitesse, 0.99 optimal qualité finale
- **Grid size** : Scaling quadratique (4×4: 100 ep, 10×10: 1000 ep)
- **Goal mobilité** : Static → Moving ≈ 10× plus de samples requis

**3. Métriques Clés** :
- **Reward** : Indicateur principal performance
- **Episode length** : Proxy efficacité (optimal ≈ Manhattan distance)
- **Loss** : Diagnostic convergence optimiseur
- **ε empirique** : Validation exploration

**4. Visualisations Critiques** :
- **Heatmaps V\*** : Indispensables pour valider apprentissage
- **Convergence curves** : Détectent plateaux/instabilités
- **GIFs** : Seule validation qualitative fiable (métriques peuvent mentir)

---

<div align="center">

<br/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&color=2ECC71&center=true&vCenter=true&width=600&lines=63+Fichiers+Catalogués+%E2%9C%85;Analyses+Complètes+%E2%9C%85;Justifications+Documentées+%E2%9C%85" alt="Typing SVG" />

<br/><br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer"/>

</div>
