"""Four decisions per episode, one pipeline evaluation at the terminal step."""
from math import lcm

import gymnasium as gym
import numpy as np

from .automl import Candidate, candidates
from .data import profile


class PipelineEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, evaluator, space):
        super().__init__()
        candidates(space)  # Validate before starting a rollout.
        self.evaluator = evaluator
        self.menus = [space[k] for k in ["scalers", "selection", "classifiers", "hyperparameter_levels"]]
        self.sizes = [len(m) for m in self.menus]
        # Modulo aliases have equal multiplicity at every stage, preserving uniform priors.
        self.action_space = gym.spaces.Discrete(lcm(*self.sizes))
        self.meta = profile(evaluator.X, evaluator.y)
        width = 7 + sum(self.sizes) + 5 + self.action_space.n + 2
        self.observation_space = gym.spaces.Box(0, 1, shape=(width,), dtype=np.float32)
        self.episodes = []
        self.stage = 4

    def _observation(self):
        choices = np.zeros(sum(self.sizes), dtype=np.float32)
        offset = 0
        for i, size in enumerate(self.sizes):
            if i < len(self.choices):
                choices[offset + self.choices[i]] = 1
            offset += size
        stage = np.eye(5, dtype=np.float32)[self.stage]
        previous = np.zeros(self.action_space.n, dtype=np.float32)
        if self.previous is not None:
            previous[self.previous] = 1
        best = self.evaluator.best
        return np.concatenate([self.meta, choices, stage, previous,
                               [len(self.evaluator.records) / self.evaluator.budget,
                                best["inner_score"] if best else 0]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        self.stage, self.choices, self.previous, self.steps = 0, [], None, []
        return self._observation(), {}

    def step(self, action):
        if self.stage == 4:
            raise RuntimeError("Reset before stepping a finished episode")
        if not self.action_space.contains(action):
            raise ValueError("Action outside the declared space")
        state = self._observation()
        self.choices.append(int(action) % self.sizes[self.stage])
        self.previous = int(action)
        self.stage += 1
        done = self.stage == 4
        reward, info = 0.0, {}
        if done:
            candidate = Candidate(*(m[c] for m, c in zip(self.menus, self.choices)))
            record = self.evaluator.evaluate(candidate)
            reward = record["inner_score"]
            info = {"evaluation": record["evaluation"], "status": record["status"]}
        next_state = self._observation()
        self.steps.append({"state": state.tolist(), "action": int(action), "base_reward": reward,
                           "shaped_reward": reward, "potential": 0.0, "next_potential": 0.0,
                           "shaping_difference": 0.0, "next_state": next_state.tolist(), "done": done})
        if done:
            self.episodes.append({"steps": self.steps.copy(), **record})
        return next_state, reward, done, False, info
