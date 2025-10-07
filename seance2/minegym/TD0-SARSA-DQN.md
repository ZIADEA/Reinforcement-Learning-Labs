TD0 interpretation des result

C est les valeur optimal mi goal est a la case 3,3; c est négatif partout a cause de comment j ai fixer les reward par défaut de mon env . 

L interprétation je sais pas mais je peut dire que plus on se rapproche de goal plus le V augment ce qui est normal ( si me me souviens d une vidéo se j avais suivit donc ça converge bien ) 


mais  la courbe des moyennes : on voit presque une moyenne comme stationnaire dans tout les épisodes .donc on converge bien des les 1er episode ver un retour = -20.

pour sarsa on vois que des 250 epsode environ il aprend bien a maximiser le retour =0 .

DQN luit evolu progressivement et on voi que le retour maximal par dqn est > a 0 et se stationnarise apres 1000episode.

l interpretation se base sir l alor des corb et les ecart de retour et nom sur les valeur exact var on remaque ve V* de sarsa et DQN son semblable mais different  et celle de TD0 est caremment differente des 2 autre . cela est du a plusieur facteur .

- Avec **TD(0) linéaire**, je fais juste de la **prédiction** de \(V^\pi\) pour **une politique fixe** (dans mon code elle est presque aléatoire). Du coup l’agent traîne, paye le coût \(-1\) à chaque pas, et ça donne des **valeurs négatives** loin du goal.

- Avec **SARSA(0)** et **DQN**, je fais du **control** : l’agent apprend une **politique qui vise le but**, et quand je trace
  \(V(s)=\max_a Q(s,a)\) j’obtiens des **valeurs positives** (récompense \(+50\) moins quelques pas).

- Les **approximateurs** ne sont pas les mêmes :
  - TD(0) : un seul vecteur de poids pour \(V(s)\) → capacité limitée ;
  - SARSA : des **poids par action** pour \(Q(s,a)\) → plus expressif ;
  - DQN : **réseau non linéaire** → encore plus puissant.
  Donc les cartes ne peuvent pas coïncider.

- Les **politiques et les données** d’apprentissage diffèrent aussi :
  TD(0) suit la politique imposée, alors que SARSA/DQN utilisent de l’ε-greedy qui devient de plus en plus “goal-oriented”.

- Et puis il y a **l’aléatoire** (seeds, exploration, tirages, etc.) qui peut ajouter des écarts.

Bref : objets d’apprentissage différents + capacités différentes + politiques/données différentes + un peu de hasard
= **des cartes \(V\) différentes**.
