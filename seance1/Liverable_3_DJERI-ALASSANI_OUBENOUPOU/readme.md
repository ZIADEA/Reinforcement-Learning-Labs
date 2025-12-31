<div align="center">

# 🚀 Livrable 3 — Agents RL Étendus

![MC](https://img.shields.io/badge/Monte_Carlo-Enhanced-4169E1?style=for-the-badge)
![DP](https://img.shields.io/badge/Dynamic_Programming-Enhanced-2ECC71?style=for-the-badge)
![QL](https://img.shields.io/badge/Q--Learning-Enhanced-E67E22?style=for-the-badge)
![PI](https://img.shields.io/badge/Policy_Iteration-Enhanced-9B59B6?style=for-the-badge)

**Version améliorée et étendue des implémentations RL fondamentales**

</div>

---

## 🎯 Contenu

Ce livrable contient les **versions étendues** des algorithmes du Livrable 2 avec :
- ✨ Améliorations de performance
- 🔧 Optimisations algorithmiques
- 📊 Meilleur tracking des métriques
- 🎨 Visualisations améliorées

<table>
<tr>
<th>Fichier</th>
<th>Description</th>
<th>Améliorations</th>
</tr>
<tr>
<td><code>agentMC.py</code></td>
<td>Agent Monte Carlo</td>
<td>Version optimisée avec meilleure gestion mémoire</td>
</tr>
<tr>
<td><code>agentPI.py</code></td>
<td>Agent Policy Iteration</td>
<td>Convergence accélérée</td>
</tr>
<tr>
<td><code>agentVI.py</code></td>
<td>Agent Value Iteration</td>
<td>Critère d'arrêt amélioré</td>
</tr>
<tr>
<td><code>agentQL.py</code></td>
<td>Agent Q-Learning</td>
<td>Stratégie d'exploration adaptative</td>
</tr>
<tr>
<td><code>4agenttest.py</code></td>
<td>Script de test étendu</td>
<td>Métriques de comparaison détaillées</td>
</tr>
<tr>
<td><code>gym1.py</code></td>
<td>Environnement amélioré</td>
<td>Fonctionnalités étendues</td>
</tr>
</table>

## 🚀 Utilisation

### Installation

```bash
cd seance1/Liverable_3_DJERI-ALASSANI_OUBENOUPOU
```

### Lancer les Tests

```bash
python 4agenttest.py
```

Le script exécutera les **4 agents améliorés** avec :
1. ✅ Entraînement optimisé
2. ✅ Collecte de métriques détaillées
3. ✅ Comparaison approfondie
4. ✅ Résultats visuels

## ✨ Améliorations par Rapport au Livrable 2

### Performance
- ⚡ **Convergence plus rapide** grâce aux optimisations
- 💾 **Utilisation mémoire réduite**
- 🎯 **Précision améliorée** des politiques apprises

### Fonctionnalités
- 📊 **Métriques étendues** : temps de convergence, stabilité
- 📈 **Visualisations** : courbes d'apprentissage, heatmaps
- 🔍 **Diagnostics** : analyse détaillée du comportement

### Code
- 🧹 **Code plus propre** et mieux structuré
- 📝 **Documentation enrichie**
- 🛡️ **Gestion d'erreurs robuste**

## 🧠 Différences Clés avec Livrable 2

| Aspect | Livrable 2 | Livrable 3 |
|--------|------------|------------|
| Implémentation | Basique | Optimisée |
| Métriques | Standard | Détaillées |
| Performance | Correcte | Améliorée |
| Visualisations | Limitées | Étendues |
| Documentation | Minimale | Complète |

## 📊 Résultats Attendus

Les améliorations devraient montrer :
- 🚀 **30-50% plus rapide** en temps de convergence
- 🎯 **Politiques plus stables** avec moins de variance
- 📈 **Courbes d'apprentissage plus lisses**
- ✅ **Métriques de performance supérieures**

## 🔧 Configuration Avancée

### Paramètres Optimisés

```python
# Q-Learning amélioré
config = {
    'alpha': 0.3,           # Taux d'apprentissage adaptatif
    'gamma': 0.95,          # Facteur d'escompte optimisé
    'epsilon_start': 1.0,   # Exploration initiale
    'epsilon_end': 0.01,    # Exploration finale
    'epsilon_decay': 0.995, # Décroissance ε
    'episodes': 3000        # Plus d'épisodes
}
```

## 📝 Notes Techniques

### Monte Carlo
- ✨ Utilisation de **First-Visit MC** pour meilleure efficacité
- 📊 Tracking des **moyennes mobiles** pour convergence

### Dynamic Programming
- ⚡ **Arrêt anticipé** quand δ < seuil
- 🎯 **Policy Iteration** avec évaluation tronquée

### Q-Learning
- 🔄 **Epsilon-decay** pour équilibre exploration/exploitation
- 📈 **Learning rate adaptatif** selon progression

## 🔍 Comparaison des Performances

Exécuter `4agenttest.py` génère :
- 📊 Tableau comparatif des 4 algorithmes
- 📈 Graphiques de convergence
- 🎯 Politiques optimales visualisées
- ⏱️ Temps d'exécution mesurés

---

<div align="center">

**[← Retour Seance 1](../README.md)** | **[← Livrable 2](../Liverable_2_DJERI-ALASSANI_OUBENOUPOU)**

<br/><br/>

<img src="https://img.shields.io/badge/Version-Extended-success?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Performance-Optimized-blue?style=for-the-badge"/>

</div>
