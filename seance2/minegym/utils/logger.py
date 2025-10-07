from typing import List, Dict, Any
import numpy as np

class StepLogger:
    def __init__(self):
        self.total_steps = 0
        self.steps_state: List[int] = []
        self.steps_action: List[int] = []
        self.steps_reward: List[float] = []
        self.steps_epsilon: List[float] = []
        self.steps_greedy: List[bool] = []
        self.steps_episode: List[int] = []
        self.steps_done: List[bool] = []

        self.episodes_return: List[float] = []
        self.episodes_length: List[int] = []

    def log(self, **kwargs):
        self.steps_state.append(kwargs["state"])
        self.steps_action.append(kwargs["action"])
        self.steps_reward.append(float(kwargs["reward"]))
        self.steps_epsilon.append(float(kwargs["epsilon"]))
        self.steps_greedy.append(bool(kwargs["greedy"]))
        self.steps_episode.append(int(kwargs["episode"]))
        self.steps_done.append(bool(kwargs["done"]))
        self.total_steps += 1

    def close_episode(self, G: float, length: int):
        self.episodes_return.append(float(G))
        self.episodes_length.append(int(length))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_state": np.array(self.steps_state),
            "step_action": np.array(self.steps_action),
            "step_reward": np.array(self.steps_reward),
            "step_epsilon": np.array(self.steps_epsilon),
            "step_greedy": np.array(self.steps_greedy, dtype=bool),
            "step_episode": np.array(self.steps_episode),
            "step_done": np.array(self.steps_done, dtype=bool),
            "ep_return": np.array(self.episodes_return),
            "ep_length": np.array(self.episodes_length),
        }
