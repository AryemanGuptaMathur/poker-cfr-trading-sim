import numpy as np
from collections import defaultdict

class Node:
    """Game tree node for infosets."""
    def __init__(self, actions):
        self.actions = actions
        self.regret_sum = np.zeros(len(actions))
        self.strategy_sum = np.zeros(len(actions))
        self.strategy = np.ones(len(actions)) / len(actions)  # Uniform init

    def get_strategy(self, realization_weight=1.0):
        """Regret-matching strategy."""
        normalizing_sum = np.sum(np.maximum(self.regret_sum, 0))
        if normalizing_sum > 0:
            self.strategy = np.maximum(self.regret_sum, 0) / normalizing_sum
        else:
            self.strategy = np.ones(len(self.actions)) / len(self.actions)
        self.strategy_sum += realization_weight * self.strategy
        return self.strategy

    def get_average_strategy(self):
        """Average strategy for equilibrium approx."""
        normalizing_sum = np.sum(self.strategy_sum)
        return self.strategy_sum / normalizing_sum if normalizing_sum > 0 else np.ones(len(self.actions)) / len(self.actions)

def chance_sample(deck):
    """Simple chance sampling for cards."""
    return np.random.choice(deck)
