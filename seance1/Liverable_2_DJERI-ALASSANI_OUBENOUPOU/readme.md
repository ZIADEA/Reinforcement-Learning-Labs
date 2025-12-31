<div align="center">

# 📖 Livrable 2 — Agents Fondamentaux RL

![MC](https://img.shields.io/badge/Monte_Carlo-4169E1?style=for-the-badge)
![DP](https://img.shields.io/badge/Dynamic_Programming-2ECC71?style=for-the-badge)
![QL](https://img.shields.io/badge/Q--Learning-E67E22?style=for-the-badge)
![PI](https://img.shields.io/badge/Policy_Iteration-9B59B6?style=for-the-badge)

**Implémentation complète des 4 algorithmes fondamentaux de Reinforcement Learning**

</div>

---

## 🎯 Contenu

Ce livrable contient les **implémentations de base** des algorithmes RL fondamentaux :

<table>
<tr>
<th>Fichier</th>
<th>Description</th>
<th>Algorithme</th>
</tr>
<tr>
<td><code>agentMC.py</code></td>
<td>Agent Monte Carlo</td>
<td>Apprentissage par épisodes complets avec moyenne des retours</td>
</tr>
<tr>
<td><code>agentPI.py</code></td>
<td>Agent Policy Iteration</td>
<td>Amélioration itérative de politique avec évaluation complète</td>
</tr>
<tr>
<td><code>agentVI.py</code></td>
<td>Agent Value Iteration</td>
<td>Convergence vers fonction de valeur optimale</td>
</tr>
<tr>
<td><code>agentQL.py</code></td>
<td>Agent Q-Learning</td>
<td>Apprentissage par différence temporelle (TD)</td>
</tr>
<tr>
<td><code>4agenttest.py</code></td>
<td>Script de test</td>
<td>Exécution et comparaison des 4 agents</td>
</tr>
<tr>
<td><code>gym1.py</code></td>
<td>Environnement</td>
<td>Environnement Gym personnalisé pour les tests</td>
</tr>
</table>

## 🚀 Utilisation

### Installation

```bash
cd seance1/Liverable_2_DJERI-ALASSANI_OUBENOUPOU
```

### Lancer les Tests

```bash
python 4agenttest.py
```

Ce script va :
1. ✅ Charger l'environnement `gym1.py`
2. ✅ Entraîner les 4 agents (MC, PI, VI, Q-Learning)
3. ✅ Comparer leurs performances
4. ✅ Afficher les résultats

## 🧠 Algorithmes Détaillés

### 🔵 Monte Carlo (MC)
- **Principe** : Apprentissage à partir d'épisodes complets
- **Méthode** : Moyenne des retours observés
- **Avantage** : Pas besoin du modèle de l'environnement
- **Fichier** : `agentMC.py`

### 🟢 Dynamic Programming (PI/VI)
- **Policy Iteration** : Évaluation puis amélioration de politique
- **Value Iteration** : Mise à jour directe vers V*
- **Avantage** : Convergence garantie
- **Fichiers** : `agentPI.py`, `agentVI.py`

### 🟠 Q-Learning
- **Principe** : Temporal Difference (TD)
- **Formule** : Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
- **Avantage** : Apprentissage en ligne, sans modèle
- **Fichier** : `agentQL.py`

## 📊 Résultats Attendus

Après exécution de `4agenttest.py`, vous devriez observer :
- ✅ Convergence de chaque algorithme
- ✅ Politiques optimales similaires
- ✅ Différences de vitesse de convergence
- ✅ Comparaison des performances

## 🔧 Paramètres Clés

| Paramètre | Description | Valeur Typique |
|-----------|-------------|----------------|
| `alpha` (α) | Taux d'apprentissage | 0.1 - 0.5 |
| `gamma` (γ) | Facteur d'escompte | 0.9 - 0.99 |
| `epsilon` (ε) | Taux d'exploration | 0.1 - 0.3 |
| `episodes` | Nombre d'épisodes | 1000 - 5000 |

## 📝 Notes

- Ce livrable représente les **implémentations de base** des algorithmes
- Voir **Livrable 3** pour la version étendue avec améliorations
- L'environnement `gym1.py` est simple pour faciliter l'apprentissage

---

<div align="center">

**[← Retour Seance 1](../README.md)** | **[Livrable 3 →](../Liverable_3_DJERI-ALASSANI_OUBENOUPOU)**

</div>
