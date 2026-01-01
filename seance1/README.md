<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,9,12&height=140&section=header&text=Seance%201&fontSize=48&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=Fondamentaux%20du%20Reinforcement%20Learning&descAlignY=55&descAlign=50"/>

<br/>

![RL Basics](https://img.shields.io/badge/RL-Fundamentals-4A90E2?style=for-the-badge&logo=python)
![Algorithms](https://img.shields.io/badge/Algorithms-4-2ecc71?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)

<br/>

**Implémentations des algorithmes fondamentaux de Reinforcement Learning**

</div>

<br/>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

## 🎯 Aperçu

Cette séance couvre les **algorithmes fondamentaux du Reinforcement Learning** à travers trois livrables progressifs qui implémentent et comparent différentes approches d'apprentissage.

### Algorithmes Implémentés

<table>
<tr>
<td align="center" width="25%">
<br/>
<img src="https://img.shields.io/badge/Monte_Carlo-4169E1?style=for-the-badge&logo=python"/>
<br/><br/>
<b>MC</b>
<br/>
Méthodes Monte Carlo
</td>
<td align="center" width="25%">
<br/>
<img src="https://img.shields.io/badge/Dynamic_Programming-2ECC71?style=for-the-badge&logo=python"/>
<br/><br/>
<b>DP</b>
<br/>
Prog. Dynamique
</td>
<td align="center" width="25%">
<br/>
<img src="https://img.shields.io/badge/Q--Learning-E67E22?style=for-the-badge&logo=python"/>
<br/><br/>
<b>Q-Learning</b>
<br/>
Temporal Difference
</td>
<td align="center" width="25%">
<br/>
<img src="https://img.shields.io/badge/Policy_Iteration-9B59B6?style=for-the-badge&logo=python"/>
<br/><br/>
<b>PI/VI</b>
<br/>
Itération Politique
</td>
</tr>
</table>

## 📦 Structure des Livrables

<div align="center">
<img src="https://progress-bar.dev/100/?title=Complétion&width=500&color=2ecc71"/>
</div>

<br/>

<table>
<tr>
<th width="20%">Livrable</th>
<th width="40%">Contenu</th>
<th width="40%">Fichiers Clés</th>
</tr>

<tr>
<td align="center">
<br/>
<img src="https://img.shields.io/badge/Livrable_1-Introduction-3498db?style=for-the-badge"/>
<br/><br/>
<a href="Liverable_1_DJERI-ALASSANI_OUBENOUPOU">📂 Voir</a>
</td>
<td>
<b>Découverte de Gymnasium</b>
<br/><br/>
Première prise en main de l'environnement Gym avec création d'un environnement simple
</td>
<td>
• <code>gym1.py</code> - Environnement de base
<br/><br/>
<img src="https://img.shields.io/badge/✓-Complete-success?style=flat-square"/>
</td>
</tr>

<tr>
<td align="center">
<br/>
<img src="https://img.shields.io/badge/Livrable_2-Core_Algorithms-e67e22?style=for-the-badge"/>
<br/><br/>
<a href="Liverable_2_DJERI-ALASSANI_OUBENOUPOU">📂 Voir</a>
</td>
<td>
<b>Agents Fondamentaux</b>
<br/><br/>
Implémentation complète des 4 algorithmes classiques de RL
</td>
<td>
• <code>agentMC.py</code> - Monte Carlo<br/>
• <code>agentPI.py</code> - Policy Iteration<br/>
• <code>agentVI.py</code> - Value Iteration<br/>
• <code>agentQL.py</code> - Q-Learning<br/>
• <code>4agenttest.py</code> - Tests comparatifs<br/>
<br/>
<img src="https://img.shields.io/badge/✓-Complete-success?style=flat-square"/>
</td>
</tr>

<tr>
<td align="center">
<br/>
<img src="https://img.shields.io/badge/Livrable_3-Extended-27ae60?style=for-the-badge"/>
<br/><br/>
<a href="Liverable_3_DJERI-ALASSANI_OUBENOUPOU">📂 Voir</a>
</td>
<td>
<b>Version Étendue</b>
<br/><br/>
Amélioration et extension des implémentations précédentes
</td>
<td>
• Mêmes agents avec améliorations<br/>
• <code>4agenttest.py</code> - Tests étendus<br/>
• Environnement <code>gym1.py</code> amélioré<br/>
<br/>
<img src="https://img.shields.io/badge/✓-Complete-success?style=flat-square"/>
</td>
</tr>

</table>

## 🚀 Démarrage Rapide

<details open>
<summary><b>⚙️ 1. Activer l'environnement</b></summary>

```powershell
& C:\Users\DJERI\VSCODE\Programmation\python\environnements\rl_venv\Scripts\Activate.ps1
```
</details>

<details>
<summary><b>▶️ 2. Tester les agents (Livrable 2 ou 3)</b></summary>

```bash
cd seance1/Liverable_2_DJERI-ALASSANI_OUBENOUPOU
python 4agenttest.py
```

Ou pour le livrable 3 :

```bash
cd seance1/Liverable_3_DJERI-ALASSANI_OUBENOUPOU
python 4agenttest.py
```
</details>

<details>
<summary><b>📊 3. Interpréter les résultats</b></summary>

Le script `4agenttest.py` exécute et compare les 4 algorithmes :
- **Monte Carlo** : Apprentissage par épisodes complets
- **Policy Iteration** : Amélioration itérative de politique
- **Value Iteration** : Convergence vers fonction de valeur optimale
- **Q-Learning** : Apprentissage par différence temporelle

</details>

## 🧠 Concepts Clés Couverts

### Programmation Dynamique
- ✅ Itération de valeur (Value Iteration)
- ✅ Itération de politique (Policy Iteration)
- ✅ Évaluation de politique

### Méthodes Monte Carlo
- ✅ Estimation par échantillonnage
- ✅ Apprentissage sans modèle
- ✅ Moyenne des retours

### Q-Learning (TD)
- ✅ Apprentissage par différence temporelle
- ✅ Mise à jour incrémentale
- ✅ Exploration vs Exploitation (ε-greedy)

## 📚 Ressources

<table>
<tr>
<td width="50%" align="center">
<br/>
📖 <b>Documentation</b>
<br/><br/>
Chaque livrable contient son README<br/>
avec instructions spécifiques
<br/><br/>
</td>
<td width="50%" align="center">
<br/>
🧪 <b>Tests</b>
<br/><br/>
Scripts <code>4agenttest.py</code><br/>
pour validation des algorithmes
<br/><br/>
</td>
</tr>
</table>

## 🔍 Navigation

| Besoin | Destination |
|--------|-------------|
| 🎓 Apprendre les bases Gym | → [Liverable 1](Liverable_1_DJERI-ALASSANI_OUBENOUPOU) |
| 🤖 Voir les implémentations | → [Liverable 2](Liverable_2_DJERI-ALASSANI_OUBENOUPOU) |
| 🚀 Version améliorée | → [Liverable 3](Liverable_3_DJERI-ALASSANI_OUBENOUPOU) |

---

<div align="center">

<br/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&color=4A90E2&center=true&vCenter=true&width=500&lines=4+Algorithmes+Impl%C3%A9ment%C3%A9s;Fondations+du+RL+%E2%9C%85" alt="Typing SVG" />

<br/><br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,9,12&height=100&section=footer"/>

</div>
