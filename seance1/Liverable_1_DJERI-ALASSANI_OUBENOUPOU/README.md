<div align="center">

# 🎓 Livrable 1 — Découverte Gymnasium

![Gym](https://img.shields.io/badge/Gymnasium-Introduction-00A67E?style=for-the-badge&logo=openai)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)

**Première prise en main de l'environnement Gymnasium**

</div>

---

## 🎯 Objectif

Ce premier livrable introduit les **bases de Gymnasium (anciennement OpenAI Gym)** avec :
- 📚 Découverte de l'API Gymnasium
- 🏗️ Création d'un environnement simple
- 🔄 Compréhension du cycle step/reset
- 🎮 Interaction basique agent-environnement

## 📦 Contenu

<table>
<tr>
<th>Fichier</th>
<th>Description</th>
</tr>
<tr>
<td><code>gym1.py</code></td>
<td>Environnement Gymnasium personnalisé de base</td>
</tr>
</table>

## 🚀 Utilisation

### Lancer l'Environnement

```bash
cd seance1/Liverable_1_DJERI-ALASSANI_OUBENOUPOU
python gym1.py
```

## 🧠 Concepts Couverts

### API Gymnasium
- ✅ **`reset()`** : Initialiser l'environnement
- ✅ **`step(action)`** : Exécuter une action
- ✅ **`render()`** : Visualiser l'état
- ✅ **Observation space** : Espace des états
- ✅ **Action space** : Espace des actions

### Structure d'un Environnement

```python
class CustomEnv(gym.Env):
    def __init__(self):
        # Définir observation_space et action_space
        pass
    
    def reset(self):
        # Réinitialiser l'environnement
        return observation
    
    def step(self, action):
        # Exécuter l'action
        return observation, reward, done, info
```

## 📚 Apprentissage

Ce livrable sert de **fondation** pour :
- Comprendre la structure MDP (Markov Decision Process)
- Découvrir le cycle interaction agent-environnement
- Préparer les implémentations d'algorithmes (Livrables 2 et 3)

## 🔍 Points Clés

| Concept | Description |
|---------|-------------|
| **État** | Configuration actuelle de l'environnement |
| **Action** | Décision prise par l'agent |
| **Récompense** | Feedback de l'environnement |
| **Épisode** | Séquence état-action jusqu'à terminaison |

## ➡️ Prochaines Étapes

Après avoir compris cet environnement de base :
1. 🔜 **Livrable 2** : Implémenter les agents (MC, PI, VI, QL)
2. 🔜 **Livrable 3** : Versions améliorées et optimisées

---

<div align="center">

**[← Retour Seance 1](../README.md)** | **[Livrable 2 →](../Liverable_2_DJERI-ALASSANI_OUBENOUPOU)**

<br/><br/>

<img src="https://img.shields.io/badge/Introduction-Gymnasium-00A67E?style=flat-square"/>

</div>
