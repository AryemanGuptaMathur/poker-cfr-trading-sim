from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from abstract_game import AbstractGame
from utils import Node, regret_matching


@dataclass
class RegretSample:
    state: np.ndarray
    regrets: np.ndarray


class RegretNet(nn.Module):
    """Small MLP used to regress instantaneous regret targets."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        final_layer = self.network[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class DeepCFRTrainer:
    """Approximate CFR regrets with a neural network trained on sampled traversals."""

    def __init__(
        self,
        game: AbstractGame,
        seed: int = 0,
        hidden_size: int = 128,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        updates_per_iteration: int = 4,
    ) -> None:
        self.game = game
        self.name = f"{game.name.replace('Vanilla CFR', '').strip()} Deep CFR".strip()
        self.seed = seed
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.batch_size = batch_size
        self.updates_per_iteration = updates_per_iteration
        self.loss_fn = nn.MSELoss()
        self.regret_net = RegretNet(game.encoding_size, game.max_actions, hidden_size=hidden_size)
        self.optimizer = torch.optim.Adam(self.regret_net.parameters(), lr=learning_rate)
        self.buffer: deque[RegretSample] = deque(maxlen=buffer_size)
        self.average_nodes: dict[str, Node] = {}

    def reset(self) -> None:
        self.buffer.clear()
        self.average_nodes.clear()
        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        self.regret_net = RegretNet(
            self.game.encoding_size,
            self.game.max_actions,
            hidden_size=self.hidden_size,
        )
        self.optimizer = torch.optim.Adam(self.regret_net.parameters(), lr=self.learning_rate)

    def average_strategy_profile(self) -> dict[str, np.ndarray]:
        return {key: node.get_average_strategy() for key, node in self.average_nodes.items()}

    def _strategy_from_network(self, state: object, player: int) -> np.ndarray:
        encoded_state = self.game.encode_state(state, player)
        state_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            raw_regrets = self.regret_net(state_tensor).squeeze(0).cpu().numpy()
        legal_actions = self.game.legal_actions(state)
        return regret_matching(raw_regrets[: len(legal_actions)])

    def _average_node(self, state: object) -> tuple[str, Node]:
        player = self.game.current_player(state)
        info_key = self.game.info_key(state, player)
        node = self.average_nodes.get(info_key)
        if node is None:
            node = Node(self.game.legal_actions(state))
            self.average_nodes[info_key] = node
        return info_key, node

    def traverse(self, state: object, traversing_player: int, reach_probs: np.ndarray) -> float:
        if self.game.is_terminal(state):
            return self.game.terminal_utility(state, traversing_player)

        current_player = self.game.current_player(state)
        legal_actions = self.game.legal_actions(state)
        strategy = self._strategy_from_network(state, current_player)
        _, average_node = self._average_node(state)
        average_node.accumulate_strategy(reach_probs[current_player], strategy)

        action_values = np.zeros(len(legal_actions), dtype=float)
        node_value = 0.0

        for action_index, action in enumerate(legal_actions):
            next_reach = reach_probs.copy()
            next_reach[current_player] *= strategy[action_index]
            action_values[action_index] = self.traverse(
                self.game.next_state(state, action),
                traversing_player,
                next_reach,
            )
            node_value += strategy[action_index] * action_values[action_index]

        if current_player == traversing_player:
            padded_regrets = np.zeros(self.game.max_actions, dtype=np.float32)
            padded_regrets[: len(legal_actions)] = action_values - node_value
            self.buffer.append(
                RegretSample(
                    state=self.game.encode_state(state, current_player),
                    regrets=padded_regrets,
                )
            )

        return node_value

    def _train_regret_network(self) -> float:
        if not self.buffer:
            return 0.0

        losses = []
        sample_count = len(self.buffer)
        for _ in range(self.updates_per_iteration):
            batch_size = min(self.batch_size, sample_count)
            indices = self.rng.choice(sample_count, size=batch_size, replace=False)
            batch = [self.buffer[int(index)] for index in indices]
            states = torch.tensor(np.stack([sample.state for sample in batch]), dtype=torch.float32)
            targets = torch.tensor(np.stack([sample.regrets for sample in batch]), dtype=torch.float32)

            predictions = self.regret_net(states)
            loss = self.loss_fn(predictions, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.item()))

        return float(np.mean(losses))

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
            "loss": [],
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
            loss = self._train_regret_network()

            should_evaluate = (
                eval_every > 0
                and (iteration == 1 or iteration % eval_every == 0 or iteration == iterations)
            )
            if should_evaluate:
                metrics["iteration"].append(float(iteration))
                metrics["utility"].append(float(cumulative_utility / iteration))
                metrics["loss"].append(float(loss))
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
