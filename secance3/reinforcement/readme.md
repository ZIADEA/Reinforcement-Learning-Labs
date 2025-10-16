# resultats d une batery de 2000 episodes et 100 test pour chaque algorithme
avec epsilon=0.05,alpha=0.2,gamma=0.8 dans le medieumclassic world : 
| scores | Value iteration | Qlearning | Qlearning approxiamtif 4features (bias,of-ghosts-1-step-away,eats-food,closest-food) |Qlearning approxiamtif 8features (bias,of-ghosts-1-step-away,eats-food,closest-food,hits-wall,towards-closest-food,ghost-dist,scared-ghost-near) |
|---|---|---|---|---|
| Average Rewards over all training | impossible | -394.07 | 966.15 |1061.27 |
| Average Rewards for last 100 episodes | impossible | -387.97 |978.13 |1089.12 |
| first score after traning | impossible | -369 | Contenu Ligne 1, Col 3 |1344 |
| last score after traning | impossible | -376 | Contenu Ligne 1, Col 3 |1336 |
| Average score  after traning | impossible |  -374.61 | Contenu Ligne 1, Col 3 |1229.82 |


NB : Bien que Pacman soit, en théorie, modélisable comme un MDP, le projet Berkeley n’expose pas Pacman via une API MDP (avec getStates, getTransitionStatesAndProbs, getReward, etc.). Par conséquent, l’algorithme d’itération de valeur ne peut pas être appliqué directement à Pacman et s’utilise sur Gridworld où le MDP est explicite.

