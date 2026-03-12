from __future__ import annotations

import numpy as np

from abstract_game import AbstractGame
from utils import Node


class ExternalSamplingMCCFRTrainer:
    """External-sampling MCCFR for small two-player zero-sum games.

    This trainer samples opponent actions along a single trajectory while fully
    expanding the traversing player's actions. In these environments, chance is
    already resolved at the root via `sample_initial_state`, so external
    sampling is a natural Monte Carlo baseline to compare against full CFR.
    """

    def __init__(self, game: AbstractGame, seed: int = 0) -> None:
        self.game = game
        self.seed = seed
        self.name = f"{game.name.replace('Vanilla CFR', '').strip()} External-Sampling MCCFR".strip()
        self.rng = np.random.default_rng(seed)
        self.infosets: dict[str, Node] = {}

    def reset(self) -> None:
        self.infosets.clear()
        self.rng = np.random.default_rng(self.seed)

    def _get_node(self, state: object) -> tuple[str, Node]:
        player = self.game.current_player(state)
        info_key = self.game.info_key(state, player)
        node = self.infosets.get(info_key)
        if node is None:
            node = Node(self.game.legal_actions(state))
            self.infosets[info_key] = node
        return info_key, node

    def average_strategy_profile(self) -> dict[str, np.ndarray]:
        return {key: node.get_average_strategy() for key, node in self.infosets.items()}

    def traverse(self, state: object, traversing_player: int, reach_probs: np.ndarray) -> float:
        if self.game.is_terminal(state):
            return self.game.terminal_utility(state, traversing_player)

        current_player = self.game.current_player(state)
        _, node = self._get_node(state)
        strategy = node.get_strategy()
        node.accumulate_strategy(reach_probs[current_player], strategy)

        if current_player != traversing_player:
            sampled_index = int(self.rng.choice(len(node.actions), p=strategy))
            next_reach = reach_probs.copy()
            next_reach[current_player] *= strategy[sampled_index]
            return self.traverse(
                self.game.next_state(state, node.actions[sampled_index]),
                traversing_player,
                next_reach,
            )

        action_values = np.zeros(len(node.actions), dtype=float)
        node_value = 0.0
        for action_index, action in enumerate(node.actions):
            next_reach = reach_probs.copy()
            next_reach[current_player] *= strategy[action_index]
            action_values[action_index] = self.traverse(
                self.game.next_state(state, action),
                traversing_player,
                next_reach,
            )
            node_value += strategy[action_index] * action_values[action_index]

        regrets = action_values - node_value
        node.update_regrets(
            opponent_reach=reach_probs[1 - current_player],
            instantaneous_regrets=regrets,
            variant=self.game.variant,
            discount=self.game.regret_discount,
        )
        return node_value

    def train(
        self,
        iterations: int = 5_000,
        eval_every: int = 500,
        eval_fn: object | None = None,
        eval_episodes: int = 300,
        reset: bool = True,
    ) -> dict[str, list[float]]:
        if reset:
            self.reset()

        metrics: dict[str, list[float]] = {
            "iteration": [],
            "utility": [],
            "exploitability": [],
        }
        cumulative_utility = 0.0

        for iteration in range(1, iterations + 1):
            root_state = self.game.sample_initial_state(self.rng)
            iteration_utility = 0.0
            for traversing_player in (0, 1):
                iteration_utility += self.traverse(
                    root_state,
                    traversing_player=traversing_player,
                    reach_probs=np.ones(2, dtype=float),
                )

            cumulative_utility += iteration_utility / 2.0

            should_evaluate = (
                eval_every > 0
                and (iteration == 1 or iteration % eval_every == 0 or iteration == iterations)
            )
            if should_evaluate:
                metrics["iteration"].append(float(iteration))
                metrics["utility"].append(float(cumulative_utility / iteration))
                if eval_fn is not None:
                    metrics["exploitability"].append(
                        float(
                            eval_fn(
                                self.game,
                                self.average_strategy_profile(),
                                num_episodes=eval_episodes,
                                seed=self.seed + iteration,
                            )
                        )
                    )

        return metrics
