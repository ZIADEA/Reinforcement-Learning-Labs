"""import numpy as np
np.random.seed(0)

world = np.zeros((5, 5))
N = world.shape[0]

goal = np.random.randint(25)
State = np.random.choice([x for x in range(25) if x != goal])

def reward(state):
    return 10 if state == goal else 0

def Action(state):
    moves = []
    # gauche
    if state % N != 0:
        moves.append(state - 1)
    # droite
    if state % N != N - 1:
        moves.append(state + 1)
    # haut
    if state - N >= 0:
        moves.append(state - N)
    # bas
    if state + N < N * N:
        moves.append(state + N)
    return np.random.choice(moves)

while State != goal:
    State = Action(State)
    R = reward(State)
    print(f"State: {State}, Reward: {R}")

print("But atteint :", State)


"""
import numpy as np
import matplotlib.pyplot as plt
import time

class GridEnv:
    def __init__(self, size=5, seed=0):
        np.random.seed(seed)
        self.size = size
        self.n_states = size * size
        self.action_space = 4
        self.state = None
        self.goal = None

    def reset(self):
        self.goal = np.random.randint(self.n_states)
        self.state = 14
        while self.state == self.goal:   # éviter que goal = 14
            self.goal = np.random.randint(self.n_states)
        return self.state

    def step(self, action):
        s = self.state
        if action == 0 and s % self.size != 0:            s -= 1  # gauche
        elif action == 1 and s % self.size != self.size - 1:  s += 1  # droite
        elif action == 2 and s - self.size >= 0:          s -= self.size
        elif action == 3 and s + self.size < self.n_states:   s += self.size
        self.state = s
        reward = 10 if s == self.goal else 0
        done = (s == self.goal)
        return s, reward, done

# ---- Simulation avec visualisation ----
env = GridEnv(seed=42)

episodes = 3
for ep in range(episodes):
    state = env.reset()
    done = False
    step_count = 0
    print(f"\n--- Épisode {ep+1} ---")
    print(f"Départ = {state}, Goal = {env.goal}")

    while not done and step_count < 50:
        grid = np.zeros((env.size, env.size))
        gx, gy = divmod(env.goal, env.size)
        sx, sy = divmod(state, env.size)
        grid[gx, gy] = 2
        grid[sx, sy] = 1

        plt.imshow(grid, cmap="coolwarm", vmin=0, vmax=2)
        plt.title(f"Episode {ep+1} - Step {step_count}")
        plt.xticks(range(env.size))
        plt.yticks(range(env.size))
        plt.pause(0.3)
        plt.clf()

        action = np.random.randint(env.action_space)
        next_state, reward, done = env.step(action)
        state = next_state
        step_count += 1

    print(f"Épisode terminé en {step_count} étapes.")

plt.close()

