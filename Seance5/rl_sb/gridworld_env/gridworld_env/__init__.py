from gymnasium.envs.registration import register

# Environnement simple : goal statique
register(
    id="GridWorldStatic-v0",
    entry_point="gridworld_env.grid_sb3_env:SB3GridEnv",
    kwargs={
        "rows": 4,
        "cols": 4,
        "obstacles": [],
        "goals": None,
        "moving_goal": False,
        "moving_mode": "random",
        "seed": 0,
    },
)

# Environnement plus difficile : goal qui bouge
register(
    id="GridWorldMoving-v0",
    entry_point="gridworld_env.grid_sb3_env:SB3GridEnv",
    kwargs={
        "rows": 4,
        "cols": 4,
        "obstacles": [],
        "goals": None,
        "moving_goal": True,
        "moving_mode": "random",
        "seed": 0,
    },
)
