<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=255,255,0,0,0&height=140&section=header&text=Seance%203&fontSize=48&fontColor=000&animation=fadeIn&fontAlignY=38&desc=Berkeley%20Pacman%20Project&descAlignY=55&descAlign=50"/>

<br/>

![Pacman](https://img.shields.io/badge/Project-Berkeley_Pacman-yellow?style=for-the-badge&logo=pacman&logoColor=black)
![Algorithm](https://img.shields.io/badge/Algorithm-Approximate_Q--Learning-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)

</div>

---

## 🎓 Note Pédagogique : Le Projet Berkeley Pacman

### 🏛️ Les Origines : CS188
Ce projet n'est pas un simple jeu. Il est issu du célèbre cours **CS188 (Introduction to Artificial Intelligence)** de l'Université de Californie à **Berkeley**. Conçu par **John DeNero** et **Dan Klein**, il est devenu la référence mondiale pour enseigner l'IA.
Pourquoi ? Parce qu'il offre une progression visuelle et intuitive : on commence par des algorithmes de recherche (A*), puis on passe aux MDPs, et enfin au Reinforcement Learning.

### 🧠 Le Défi : L'Explosion Combinatoire
Dans la Séance 2 (GridWorld), nous utilisions un tableau Q (Q-Table) pour stocker la valeur de chaque case.
Dans Pacman, c'est impossible. Pourquoi ?
*   L'état n'est pas juste la position de Pacman (x, y).
*   L'état = (Pos Pacman, Pos Fantôme 1, Pos Fantôme 2, ..., **État de chaque gomme**).
*   S'il y a 30 gommes (food dots), chacune peut être mangée ou non ($2^{30}$ possibilités).
*   L'espace d'états est astronomique ($> 10^{20}$). Aucun ordinateur ne peut stocker un tableau Q de cette taille.

### 💡 La Solution : Approximate Q-Learning
C'est ici que nous introduisons un concept fondamental du RL moderne : l'**Approximation de Fonction**.
Au lieu d'apprendre une valeur pour chaque état précis (ce qu'on ne reverra jamais deux fois exactement pareil), l'agent apprend à reconnaître des **caractéristiques (features)** :
1.  "Suis-je proche d'un fantôme ?" (Danger)
2.  "Suis-je proche d'une gomme ?" (Récompense)
3.  "Est-ce que je vais dans un cul-de-sac ?"

L'agent apprend des **poids** ($w$) pour ces caractéristiques.
$$ Q(s, a) = w_1 \cdot f_1(s,a) + w_2 \cdot f_2(s,a) + ... $$
C'est ce qui permet à Pacman de généraliser : s'il apprend que "Fantôme proche = Mauvais" dans le coin gauche, il saura que c'est aussi mauvais dans le coin droit.

> **📚 Référence Incontournable :**
> *DeNero, J., & Klein, D. (2010). Teaching introductory artificial intelligence with Pac-Man. In Proceedings of the Symposium on Educational Advances in Artificial Intelligence (EAAI).*

---

# resultats d une batery de 2000 episodes et 100 test pour chaque algorithme

![alt text](image.png)

avec epsilon=0.05,alpha=0.2,gamma=0.8 dans le medieumclassic world : 
| scores | Value iteration | Qlearning | Qlearning approxiamtif 4features (bias,of-ghosts-1-step-away,eats-food,closest-food) |Qlearning approxiamtif 8features (bias,of-ghosts-1-step-away,eats-food,closest-food,hits-wall,towards-closest-food,ghost-dist,scared-ghost-near) |
|---|---|---|---|---|
| Average Rewards over all training | impossible | -394.07 | 989.07 |1061.27 |
| Average Rewards for last 100 episodes | impossible | -387.97 |958.87 |1089.12 |
| first score after traning | impossible | -369 | 1322 |1344 |
| last score after traning | impossible | -376 | 1332 |1336 |
| Average score  after traning | impossible |  -374.61 | 1224.82 |1229.82 |
<div align="center">
    <img src="pacman_game.gif" alt="Pacman Game Animation" />
</div>

NB : Bien que Pacman soit, en théorie, modélisable comme un MDP, le projet Berkeley n’expose pas Pacman via une API MDP (avec getStates, getTransitionStatesAndProbs, getReward, etc.). Par conséquent, l’algorithme d’itération de valeur ne peut pas être appliqué directement à Pacman et s’utilise sur Gridworld où le MDP est explicite.

