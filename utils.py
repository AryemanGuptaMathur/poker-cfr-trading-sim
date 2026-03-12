from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def regret_matching(regrets: np.ndarray) -> np.ndarray:
    """Convert cumulative regrets into a behavior strategy."""
    positive_regrets = np.maximum(regrets, 0.0)
    normalizer = float(np.sum(positive_regrets))
    if normalizer > 0.0:
        return positive_regrets / normalizer
    return np.full(regrets.shape, 1.0 / regrets.size, dtype=float)


def sample_action(actions: Sequence[str], probabilities: np.ndarray, rng: np.random.Generator) -> str:
    """Sample a single action from a probability vector."""
    index = int(rng.choice(len(actions), p=probabilities))
    return str(actions[index])


@dataclass
class Node:
    """Tabular infoset node used by regret matching.

    Regret is tracked per action. Strategy is proportional to positive regrets,
    so actions that outperform the node value receive more probability mass.
    """

    actions: Sequence[str]
    regret_sum: np.ndarray = field(init=False)
    strategy_sum: np.ndarray = field(init=False)
    last_strategy: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        num_actions = len(self.actions)
        self.regret_sum = np.zeros(num_actions, dtype=float)
        self.strategy_sum = np.zeros(num_actions, dtype=float)
        self.last_strategy = np.full(num_actions, 1.0 / num_actions, dtype=float)

    def get_strategy(self) -> np.ndarray:
        self.last_strategy = regret_matching(self.regret_sum)
        return self.last_strategy

    def accumulate_strategy(self, realization_weight: float, strategy: np.ndarray) -> None:
        self.strategy_sum += realization_weight * strategy

    def update_regrets(
        self,
        opponent_reach: float,
        instantaneous_regrets: np.ndarray,
        variant: str = "vanilla",
        discount: float = 1.0,
    ) -> None:
        if discount != 1.0:
            self.regret_sum *= discount
        self.regret_sum += opponent_reach * instantaneous_regrets
        if variant == "cfr+":
            self.regret_sum = np.maximum(self.regret_sum, 0.0)

    def get_average_strategy(self) -> np.ndarray:
        normalizer = float(np.sum(self.strategy_sum))
        if normalizer > 0.0:
            return self.strategy_sum / normalizer
        return np.full(len(self.actions), 1.0 / len(self.actions), dtype=float)
