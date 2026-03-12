from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from utils import Node, sample_action


class AbstractGame(ABC):
    """Shared CFR implementation for small two-player zero-sum games."""

    def __init__(
        self,
        name: str,
        all_actions: list[str] | tuple[str, ...],
        seed: int = 0,
        variant: str = "vanilla",
        regret_discount: float = 1.0,
    ) -> None:
        self.name = name
        self.all_actions = tuple(all_actions)
        self.max_actions = len(self.all_actions)
        self.seed = seed
        self.variant = variant
        self.regret_discount = regret_discount
        self.rng = np.random.default_rng(seed)
        self.infosets: dict[str, Node] = {}
        self.encoding_size = 0

    def reset(self) -> None:
        self.infosets.clear()
        self.rng = np.random.default_rng(self.seed)

    @staticmethod
    def history_to_string(tokens: tuple[str, ...]) -> str:
        return "_".join(tokens) if tokens else "start"

    @abstractmethod
    def sample_initial_state(self, rng: np.random.Generator) -> Any:
        """Sample a single chance-resolved root state."""

    @abstractmethod
    def is_terminal(self, state: Any) -> bool:
        """Return True when the current state is terminal."""

    @abstractmethod
    def current_player(self, state: Any) -> int:
        """Return the id of the player to act."""

    @abstractmethod
    def legal_actions(self, state: Any) -> list[str]:
        """Return all legal actions at the current state."""

    @abstractmethod
    def next_state(self, state: Any, action: str) -> Any:
        """Advance the game by one player action."""

    @abstractmethod
    def info_key(self, state: Any, player: int) -> str:
        """Return the imperfect-information key for a player."""

    @abstractmethod
    def terminal_utility(self, state: Any, player: int) -> float:
        """Return utility from one player's perspective."""

    @abstractmethod
    def encode_state(self, state: Any, player: int) -> np.ndarray:
        """Encode an infoset as a fixed-length feature vector."""

    def enumerate_initial_states(self) -> list[tuple[Any, float]] | None:
        """Return exact chance-resolved root states when enumeration is feasible."""
        return None

    def _get_node(self, state: Any) -> tuple[str, Node]:
        player = self.current_player(state)
        info_key = self.info_key(state, player)
        actions = self.legal_actions(state)
        node = self.infosets.get(info_key)
        if node is None:
            node = Node(actions)
            self.infosets[info_key] = node
        return info_key, node

    def average_strategy_profile(self) -> dict[str, np.ndarray]:
        return {key: node.get_average_strategy() for key, node in self.infosets.items()}

    def cfr(self, state: Any, traversing_player: int, reach_probs: np.ndarray) -> float:
        """Run one full-tree CFR traversal from a chance-resolved root.

        The value returned is always from the traversing player's perspective.
        Regret is updated with `action_value - node_value`, weighted by the
        opponent reach probability. Strategy averages are weighted by the
        acting player's realization probability.
        """

        if self.is_terminal(state):
            return self.terminal_utility(state, traversing_player)

        current_player = self.current_player(state)
        _, node = self._get_node(state)
        strategy = node.get_strategy()
        node.accumulate_strategy(reach_probs[current_player], strategy)

        action_values = np.zeros(len(node.actions), dtype=float)
        node_value = 0.0

        for action_index, action in enumerate(node.actions):
            next_reach = reach_probs.copy()
            next_reach[current_player] *= strategy[action_index]
            action_values[action_index] = self.cfr(
                self.next_state(state, action),
                traversing_player,
                next_reach,
            )
            node_value += strategy[action_index] * action_values[action_index]

        if current_player == traversing_player:
            regrets = action_values - node_value
            node.update_regrets(
                opponent_reach=reach_probs[1 - current_player],
                instantaneous_regrets=regrets,
                variant=self.variant,
                discount=self.regret_discount,
            )

        return node_value

    def train(
        self,
        iterations: int = 5_000,
        eval_every: int = 500,
        eval_fn: Any | None = None,
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
            root_state = self.sample_initial_state(self.rng)
            iteration_utility = 0.0

            # Full CFR traverses once per player each iteration.
            for traversing_player in (0, 1):
                iteration_utility += self.cfr(
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
                                self,
                                self.average_strategy_profile(),
                                num_episodes=eval_episodes,
                                seed=self.seed + iteration,
                            )
                        )
                    )

        return metrics

    def _action_probabilities(
        self,
        strategy_profile: dict[str, np.ndarray],
        state: Any,
        player: int,
    ) -> tuple[list[str], np.ndarray]:
        actions = self.legal_actions(state)
        info_key = self.info_key(state, player)
        if info_key not in strategy_profile:
            probabilities = np.full(len(actions), 1.0 / len(actions), dtype=float)
            return actions, probabilities

        raw = np.asarray(strategy_profile[info_key], dtype=float)
        probabilities = raw[: len(actions)]
        total = float(np.sum(probabilities))
        if total <= 0.0:
            probabilities = np.full(len(actions), 1.0 / len(actions), dtype=float)
        else:
            probabilities = probabilities / total
        return actions, probabilities

    def policy_value_from_state(
        self,
        state: Any,
        strategy_profile: dict[str, np.ndarray],
        player: int,
    ) -> float:
        if self.is_terminal(state):
            return self.terminal_utility(state, player)

        current_player = self.current_player(state)
        actions, probabilities = self._action_probabilities(strategy_profile, state, current_player)
        value = 0.0
        for action, probability in zip(actions, probabilities):
            value += probability * self.policy_value_from_state(
                self.next_state(state, action),
                strategy_profile,
                player,
            )
        return value

    def best_response_value_from_state(
        self,
        state: Any,
        strategy_profile: dict[str, np.ndarray],
        br_player: int,
    ) -> float:
        if self.is_terminal(state):
            return self.terminal_utility(state, br_player)

        current_player = self.current_player(state)
        actions = self.legal_actions(state)

        if current_player == br_player:
            return max(
                self.best_response_value_from_state(
                    self.next_state(state, action),
                    strategy_profile,
                    br_player,
                )
                for action in actions
            )

        _, probabilities = self._action_probabilities(strategy_profile, state, current_player)
        value = 0.0
        for action, probability in zip(actions, probabilities):
            value += probability * self.best_response_value_from_state(
                self.next_state(state, action),
                strategy_profile,
                br_player,
            )
        return value

    def play_episode(
        self,
        strategy_profile: dict[str, np.ndarray],
        seed: int | None = None,
    ) -> Any:
        rng = np.random.default_rng(self.seed if seed is None else seed)
        state = self.sample_initial_state(rng)

        while not self.is_terminal(state):
            player = self.current_player(state)
            actions, probabilities = self._action_probabilities(strategy_profile, state, player)
            action = sample_action(actions, probabilities, rng)
            state = self.next_state(state, action)

        return state

    def play_with_br(
        self,
        strategy_profile: dict[str, np.ndarray],
        br_player: int,
        seed: int | None = None,
    ) -> Any:
        """Simulate one episode where one player plays a best response."""

        rng = np.random.default_rng(self.seed if seed is None else seed)
        state = self.sample_initial_state(rng)

        while not self.is_terminal(state):
            player = self.current_player(state)
            actions = self.legal_actions(state)

            if player == br_player:
                values = [
                    self.best_response_value_from_state(
                        self.next_state(state, action),
                        strategy_profile,
                        br_player,
                    )
                    for action in actions
                ]
                action = actions[int(np.argmax(values))]
            else:
                _, probabilities = self._action_probabilities(strategy_profile, state, player)
                action = sample_action(actions, probabilities, rng)

            state = self.next_state(state, action)

        return state
