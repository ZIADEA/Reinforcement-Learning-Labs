import numpy as np
import matplotlib.pyplot as plt
from gym1 import GridEnv
from agentVI import ValueIterationAgent
from agentPI import PolicyIterationAgent
from agentMC import MonteCarloAgent
from agentQL import QLearningAgent

env = GridEnv(seed=42)

# -------- VALUE ITERATION ----------
vi = ValueIterationAgent(env)
vi_history = vi.train()  # on modifie légèrement train() pour qu’elle renvoie l’évolution de V
# vi_history = [moyenne des V après chaque itération]

# -------- POLICY ITERATION ----------
pi = PolicyIterationAgent(env)
pi_history = pi.train()  # idem → renvoyer évolution de V

# -------- MONTE CARLO ----------
mc = MonteCarloAgent(env)
mc_rewards = mc.train(episodes=500)  # renvoyer reward total par épisode

# -------- Q-LEARNING ----------
ql = QLearningAgent(env)
ql_rewards = ql.train(episodes=500)  # idem

# --- PLOT ---
plt.figure(figsize=(12,5))

# Valeurs d'état (Value Function)
plt.subplot(1,2,1)
plt.plot(vi_history, label='VI - moyenne V')
plt.plot(pi_history, label='PI - moyenne V')
plt.xlabel('Itérations')
plt.ylabel('Valeur moyenne des états')
plt.title('Convergence de la fonction de valeur')
plt.legend()

# Récompenses
plt.subplot(1,2,2)
plt.plot(np.convolve(mc_rewards, np.ones(10)/10, mode='valid'), label='MC moyenné')
plt.plot(np.convolve(ql_rewards, np.ones(10)/10, mode='valid'), label='QL moyenné')
plt.xlabel('Épisodes')
plt.ylabel('Récompense totale')
plt.title('Récompenses par épisode')
plt.legend()

plt.tight_layout()
plt.show()
