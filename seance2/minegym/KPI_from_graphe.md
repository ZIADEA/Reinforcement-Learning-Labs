# Environnement et Analyse RL

## Environnement de Base (utilisé pour les figures)

### Configuration
- **Grille** : 6×8
- **Start** : état 0 (haut-gauche)
- **Goal(s)** : 1 goal à l’état 46 (par défaut). Le code supporte plusieurs goals.
- **Obstacles** : [9, 10, 11, 12, 20, 28, 36, 37]
- **Actions (déterministes)** :
  - 0=←
  - 1=→
  - 2=↑
  - 3=↓
- **Récompenses** :
  - reward_step = -1 (coût par pas)
  - reward_wall_bump = -5 (choc mur, on reste sur place)
  - reward_obstacle_attempt = -10 (tentative d’entrer dans un obstacle)
  - reward_goal = +50 (fin d’épisode)
- **Goal mobile (option)** :
  - moving_goal=False (par défaut, MDP stationnaire).
  - moving_goal=True + moving_mode="cyclic" ou "random" : le(s) goal(s) se déplacent après chaque step (en évitant obstacles, agent et doublons).
  - **Effet attendu** : apprentissage plus instable (non-stationnaire), proportion greedy plus ondulée.

### Paramètres RL
- **Agent** : Q-Learning tabulaire
- **Hyperparamètres** :
  - gamma=0.9, alpha=0.2, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01
- **Épisodes** : EPISODES=200, pas max : MAX_STEPS=50
- **Évaluation** : toutes les 50 itérations, moyenne de 20 épisodes greedy ; sauvegarde du best.

### Visualisation en Temps Réel
- **Gauche** : grille (blanc=libre, noir=obstacle, vert=goal, rouge=agent, quadrillage visible).
- **Haut-droite** : cumul du reward du step courant.
- **Bas-droite** : reward par épisode + moyenne mobile(10).

### Fichiers Générés
- `live_training.mp4` : animation du live (H.264, frame-skip & downscale pour économiser RAM).
- `V_star_heatmap_annotated.png` : heatmap annotée de V* (= max_a Q(s,a)).
- `pi_star_grid.png` : politique greedy (flèches argmax_a Q(s,a)).
- `policy_value.png` : récap valeur/politique.
- `visits.png` : heatmap des visites d’états.
- `dominant_actions.png` : action dominante observée par état.
- `summary_dashboard.png` : dashboard récapitulatif (voir calculs).
- `live_explore_exploit_empirical.png` : Exploration/Exploitation : empirique vs théorique.

### Comment chaque figure est calculée
Pendant l’entraînement on journalise à chaque step :
- **step_state**, **step_action**, **step_reward**, **step_epsilon**, **step_greedy(0/1)**, **step_episode**
et à chaque épisode :
- **ep_return** (somme des rewards), **ep_length** (nb de pas).

#### summary_dashboard.png
- **Distribution des récompenses par action** : histogrammes de step_reward séparés par action.
- **Convergence (retour par épisode)** : trace ep_return (brut) + MA(50).
- **Longueur des épisodes** : MA(50) de ep_length.
- **Distribution des actions** : histogramme de step_action.
- **Exploration vs Exploitation** :
  - **Théorique** : 1 - ε_t avec ε_t = max(ε0·decay^t, ε_min) (t = épisode).
  - **Empirique** : pour chaque épisode k, mean(step_greedy[episode==k]), lissée MA(50).
- **Récompense moyenne par pas** : MA(200) de step_reward (ordre temporel).

#### live_explore_exploit_empirical.png
- **Courbe empirique** : proportion d’actions greedy par épisode (moyenne de step_greedy), lissée MA(50).
- **Courbe théorique** : 1-ε (même planning que l’agent).
- **Lecture** : plus la courbe empirique colle à la théorique, plus l’agent suit le planning d’exploration. Avec goal mobile, attendez-vous à des oscillations et une montée plus lente.

### Remarques “goal mobile”
Activez-le en instanciant l’env avec **GridEnv(..., moving_goal=True, moving_mode="cyclic" | "random")**.

Cela rend l’environnement non-stationnaire : courbes de retour plus ondulées, proportion greedy qui fluctue davantage, V* et π* reflètent la Q-table au moment final (pas un optimum stationnaire).

---

## Variation de Gamma

### Environnement (fixe)
- **Grille** : 10 × 10 (100 états).
- **Actions (tabulaires, déterministes)** :
  - 0=←
  - 1=→
  - 2=↑
  - 3=↓
- **Récompenses** :
  - reward_step = -1 (coût par pas)
  - reward_goal = +50 (arrêt de l’épisode)
  - reward_wall_bump = -5 (choc mur, on reste sur place)
  - reward_obstacle_attempt = -10 (tentative d’entrer dans un obstacle, on reste sur place)
- **Start** : case 0 (haut-gauche).
- **Goal** : case 99 (bas-droite).
- **Obstacles** : 9 cases sur la sous-diagonale (i*cols+(i-1) pour i=1..9).
- **Goals mobiles** : moving_goal = False (MDP stationnaire).
- **Taille d’épisode** : MAX_STEPS = 250.

### Paramètres RL
- **Agent** : Q-Learning tabulaire.
- **Horizon (γ)** : 11 valeurs testées : 0.0, 0.1, 0.2, …, 0.9, 0.99.
- **Exploration** : ε-greedy avec ε0=1.0, ε_decay=0.995, ε_min=0.01.
- **Alpha** : 0.20.
- **N épisodes** : EPISODES = 2000.
- **Seeds** : pour chaque γ, 5 seeds {0,1,2,3,4} (moyennes inter-seeds pour stabiliser les résultats).

### Comment chaque figure est calculée
#### 1) Par seed (répété pour 5 seeds : 0,1,2,3,4)
- L’agent exécute N = 2000 épisodes (max 250 steps chacun).
- À chaque épisode, on calcule et stocke le retour total ep_return (somme des récompenses jusqu’au done ou 250 pas).
- On obtient donc, par seed, un vecteur : ep_return_seed de taille 2000.

#### 2) Au niveau d’un γ donné (agrégation intra-γ)
- Pour un γ fixé, on a 5 vecteurs ep_return_seed (un par seed). On fait :
  - Moyenne inter-seeds par épisode : mean_ep_return[ep] = moyenne_des_ep_return_seed_sur_les_5_seeds[ep].
  - Optionnel : IC95% : IC ≈ 1.96 * std / sqrt(5) épisode-par-épisode.
  - Lissage (moyenne mobile) : on lisse mean_ep_return par une MA fenêtre 50 pour la lisibilité : MA_50(mean_ep_return).
- C’est cette courbe lissée (une par γ) qu’on trace dans “Convergence (retour moyen ± IC)”.

#### 3) Time-to-threshold (par γ)
- Pour chaque seed, on applique la MA(50) sur ep_return_seed.
- On cherche le premier épisode où la MA(50) dépasse le seuil THRESHOLD = -10 (choix expérimental adapté aux barèmes ci-dessus).
- On obtient 5 temps (un par seed).
- On affiche la moyenne ± écart-type inter-seeds dans “Temps pour atteindre le seuil”.
  - Lecture : barre plus basse = convergence plus rapide.

#### 4) Performance finale (par γ)
- Pour chaque seed, on prend la moyenne des 200 derniers épisodes de ep_return_seed.
- On obtient 5 valeurs (une par seed).
- On trace la moyenne ± écart-type inter-seeds dans “Performance finale (200 derniers épisodes)”.

#### 5) Longueur d’épisode (illustratif)
- Pour ne pas surcharger, on trace une courbe par γ pour un seed fixe (seed=0) :
  - On prend ep_length (nombre de pas par épisode) sur 2000 épisodes,
  - On lisse avec MA(50),
  - On superpose les courbes dans “Longueur des épisodes (MA 50) — par γ”.
  - Lecture : si la courbe descend et reste basse, l’agent atteint plus vite le but.

#### 6) Proportion d’actions greedy — sous-figures par γ
- **Fichier** : figures/sensitivity_gamma/sensitivity_gamma_prop_greedy_subplots.png
- **Ce que c’est** : 11 sous-graphes (un par γ) montrant, au fil des épisodes, la proportion d’actions greedy réellement prises (actions argmax(Q) vs actions d’exploration).
- **Comment c’est calculé (fidèle aux runs)** :
  - Pour chaque (γ, seed), pendant l’entraînement on marque à chaque step si l’action est greedy (step_greedy ∈ {0,1}).
  - On agrège par épisode → proportion greedy de l’épisode = moyenne de step_greedy sur ses steps.
  - Pour un γ donné, on moyennise sur les 5 seeds épisode par épisode (moyenne inter-seeds).
  - On lisse cette moyenne par MA(50).
- **Axes & lecture** : x = épisodes (0 → 2000), y = proportion greedy (0 → 1).
  - Une courbe qui monte vers 1 indique la transition exploration → exploitation.
- **Interprétation rapide** :
  - γ bas : valorise le court terme → greedy peut monter vite mais politique parfois moins “globale”.
  - γ élevé : valorise le long terme → montée plus tardive mais potentiellement plus stable/efficace une fois convergé.
  - Oscillations : exploration résiduelle (ε non nul) et/ou politique encore instable.

---

## Sensibilité à la Taille de la Grille

### Environnement (variable)
- **Tailles** : n × n avec n ∈ {4,…,10} (7 tailles).
- **Actions (tabulaires, déterministes)** :
  - 0=←
  - 1=→
  - 2=↑
  - 3=↓
- **Récompenses** :
  - reward_step = -1 (coût par pas)
  - reward_goal = +50 (arrêt de l’épisode)
  - reward_wall_bump = -5 (choc mur)
  - reward_obstacle_attempt = -10 (tentative d’entrer dans un obstacle)
- **Start** : case 0 (haut-gauche).
- **Goal** : bas-droite.
- **Obstacles** : ~10% des cases (tirage aléatoire fixe par taille pour la reproductibilité).
- **Goals mobiles** : moving_goal = False (MDP stationnaire).
- **Taille d’épisode** : MAX_STEPS = 250.

### Paramètres RL
- **Agent** : Q-Learning tabulaire (γ=0.90, α=0.20).
- **Exploration** : ε-greedy (ε0=1.0, ε_decay=0.995, ε_min=0.01).
- **N épisodes** : EPISODES = 2000.
- **Seed** : seed=42 (identique pour toutes les tailles dans ce script).

### Comment chaque figure est obtenue
#### 1) Convergence (retour moyen)
- Pendant l’entraînement, on collecte ep_return (somme des récompenses par épisode).
- On trace MA_50(ep_return) pour chaque taille.

#### 2) Performance finale (200 derniers)
- Pour chaque taille : moyenne et écart-type des 200 derniers ep_return.
- Bar chart avec barres d’erreur.

#### 3) Longueur d’épisode (MA=50)
- Pour chaque taille : on récupère ep_length (nb de pas par épisode), on lisse (MA=50) et on superpose.
- Courbe plus basse ➜ l’agent atteint plus vite le but.

#### 4) Exploration/Exploitation — théorique
- ε_t = max(ε0·decay^t, ε_min) montré en aire empilée : exploration (ε) vs exploitation (1−ε).
- Identique pour toutes les tailles (même planning ε).

#### 5) Exploration/Exploitation — empirique
- Pendant l’apprentissage, à chaque step on logge step_greedy ∈ {0,1} (si action = argmax(Q) ou non).
- Pour chaque épisode : proportion greedy = moyenne de step_greedy sur les steps de l’épisode.
- On trace, par taille, la courbe (lissée MA=50) dans 7 sous-figures.
- Lecture : montée vers 1.0 = transition vers l’exploitation; oscillations = exploration résiduelle / instabilité.