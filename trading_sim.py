from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np

from abstract_game import AbstractGame


ACTION_VOCAB = ("<pad>", "narrow", "wide", "skewbid", "skewask")
MOVE_VOCAB = ("<pad>", "moveup", "moveflat", "movedown")
ACTION_PATTERN = re.compile(r"r(\d+)-p([01])-(.+)")


@dataclass(frozen=True)
class MarketScenario:
    order_flow: tuple[int, ...]
    price_moves: tuple[float, ...]
    fill_draws: tuple[tuple[tuple[float, float], tuple[float, float]], ...]


@dataclass(frozen=True)
class TradingState:
    scenario: MarketScenario
    history: tuple[str, ...]
    round_index: int
    player_to_act: int
    pending_actions: tuple[str | None, str | None]
    positions: tuple[int, int]
    cash: tuple[float, float]
    mid_price: float
    last_move: float


class MarketMakingGame(AbstractGame):
    """Toy market-making game with synthetic order-flow scenarios."""

    def __init__(
        self,
        seed: int = 0,
        variant: str = "vanilla",
        horizon: int = 3,
        sigma: float = 0.35,
        num_scenarios: int = 512,
    ) -> None:
        variant_suffix = "CFR+" if variant == "cfr+" else "Vanilla CFR"
        super().__init__(
            name=f"Trading Sim {variant_suffix}",
            all_actions=["narrow", "wide", "skewbid", "skewask"],
            seed=seed,
            variant=variant,
        )
        self.horizon = horizon
        self.sigma = sigma
        self.market_library = self.generate_synthetic_market_data(
            num_paths=num_scenarios,
            horizon=horizon,
            seed=seed,
        )
        self.quote_params = {
            "narrow": {"bid_edge": 0.50, "ask_edge": 0.50, "bid_base": 0.62, "ask_base": 0.62},
            "wide": {"bid_edge": 1.00, "ask_edge": 1.00, "bid_base": 0.28, "ask_base": 0.28},
            "skewbid": {"bid_edge": 0.40, "ask_edge": 0.90, "bid_base": 0.74, "ask_base": 0.22},
            "skewask": {"bid_edge": 0.90, "ask_edge": 0.40, "bid_base": 0.22, "ask_base": 0.74},
        }
        self.inventory_buckets = tuple(range(-horizon, horizon + 1))
        self.action_index = {token: index for index, token in enumerate(ACTION_VOCAB)}
        self.move_index = {token: index for index, token in enumerate(MOVE_VOCAB)}
        self.encoding_size = (
            2
            + (self.horizon + 1)
            + len(self.inventory_buckets)
            + 3
            + self.horizon * (len(ACTION_VOCAB) + len(ACTION_VOCAB) + len(MOVE_VOCAB))
        )

    @staticmethod
    def generate_synthetic_market_data(
        num_paths: int,
        horizon: int,
        seed: int,
    ) -> list[MarketScenario]:
        """Generate small order-book scenarios from random walks and latent flow."""
        rng = np.random.default_rng(seed)
        scenarios: list[MarketScenario] = []

        for _ in range(num_paths):
            latent_flow = 0.0
            order_flow = []
            price_moves = []
            fill_draws = []

            for _ in range(horizon):
                latent_flow = 0.65 * latent_flow + rng.normal(0.0, 0.9)
                noisy_signal = latent_flow + rng.normal(0.0, 0.3)
                flow = int(np.clip(np.sign(noisy_signal), -1, 1))
                move = float(np.round(0.45 * flow + rng.normal(0.0, 0.35), 2))
                player_draws = (
                    (float(rng.random()), float(rng.random())),
                    (float(rng.random()), float(rng.random())),
                )
                order_flow.append(flow)
                price_moves.append(move)
                fill_draws.append(player_draws)

            scenarios.append(
                MarketScenario(
                    order_flow=tuple(order_flow),
                    price_moves=tuple(price_moves),
                    fill_draws=tuple(fill_draws),
                )
            )

        return scenarios

    def sample_initial_state(self, rng: np.random.Generator) -> TradingState:
        scenario = self.market_library[int(rng.integers(len(self.market_library)))]
        return TradingState(
            scenario=scenario,
            history=tuple(),
            round_index=0,
            player_to_act=0,
            pending_actions=(None, None),
            positions=(0, 0),
            cash=(0.0, 0.0),
            mid_price=100.0,
            last_move=0.0,
        )

    def is_terminal(self, state: TradingState) -> bool:
        return state.round_index >= self.horizon

    def current_player(self, state: TradingState) -> int:
        return state.player_to_act

    def legal_actions(self, state: TradingState) -> list[str]:
        return list(self.all_actions)

    def next_state(self, state: TradingState, action: str) -> TradingState:
        history = state.history + (self._action_token(state.round_index, state.player_to_act, action),)

        if state.player_to_act == 0:
            return TradingState(
                scenario=state.scenario,
                history=history,
                round_index=state.round_index,
                player_to_act=1,
                pending_actions=(action, None),
                positions=state.positions,
                cash=state.cash,
                mid_price=state.mid_price,
                last_move=state.last_move,
            )

        p0_action = state.pending_actions[0]
        assert p0_action is not None
        positions, cash, next_mid_price, move = self._settle_round(
            state=state,
            actions=(p0_action, action),
        )

        return TradingState(
            scenario=state.scenario,
            history=history + (self._move_token(move),),
            round_index=state.round_index + 1,
            player_to_act=0,
            pending_actions=(None, None),
            positions=positions,
            cash=cash,
            mid_price=next_mid_price,
            last_move=move,
        )

    def info_key(self, state: TradingState, player: int) -> str:
        last_move_token = self._move_token(state.last_move)
        public_history = self.history_to_string(state.history)
        return (
            f"round{state.round_index}_p{player}_inv{state.positions[player]}_"
            f"{last_move_token}_{public_history}"
        )

    def terminal_utility(self, state: TradingState, player: int) -> float:
        scores = []
        for index in (0, 1):
            mark_to_market = state.cash[index] + state.positions[index] * state.mid_price
            inventory_penalty = self.sigma * math.sqrt(abs(state.positions[index]))
            scores.append(mark_to_market - inventory_penalty)

        relative_value = scores[0] - scores[1]
        return float(relative_value if player == 0 else -relative_value)

    def encode_state(self, state: TradingState, player: int) -> np.ndarray:
        vector = np.zeros(self.encoding_size, dtype=np.float32)
        cursor = 0

        vector[cursor + player] = 1.0
        cursor += 2

        vector[cursor + state.round_index] = 1.0
        cursor += self.horizon + 1

        inventory = int(np.clip(state.positions[player], self.inventory_buckets[0], self.inventory_buckets[-1]))
        vector[cursor + (inventory - self.inventory_buckets[0])] = 1.0
        cursor += len(self.inventory_buckets)

        move_token = self._move_token(state.last_move)
        move_slot = {"moveup": 0, "moveflat": 1, "movedown": 2}[move_token]
        vector[cursor + move_slot] = 1.0
        cursor += 3

        p0_actions = ["<pad>"] * self.horizon
        p1_actions = ["<pad>"] * self.horizon
        moves = ["<pad>"] * self.horizon
        move_index = 0

        for token in state.history:
            match = ACTION_PATTERN.fullmatch(token)
            if match:
                round_index = int(match.group(1))
                acting_player = int(match.group(2))
                action_name = match.group(3)
                if acting_player == 0:
                    p0_actions[round_index] = action_name
                else:
                    p1_actions[round_index] = action_name
            elif token.startswith("move") and move_index < self.horizon:
                moves[move_index] = token
                move_index += 1

        for round_index in range(self.horizon):
            vector[cursor + self.action_index[p0_actions[round_index]]] = 1.0
            cursor += len(ACTION_VOCAB)
            vector[cursor + self.action_index[p1_actions[round_index]]] = 1.0
            cursor += len(ACTION_VOCAB)
            vector[cursor + self.move_index[moves[round_index]]] = 1.0
            cursor += len(MOVE_VOCAB)

        return vector

    def _settle_round(
        self,
        state: TradingState,
        actions: tuple[str, str],
    ) -> tuple[tuple[int, int], tuple[float, float], float, float]:
        round_index = state.round_index
        flow = state.scenario.order_flow[round_index]
        move = state.scenario.price_moves[round_index]
        draw_row = state.scenario.fill_draws[round_index]
        positions = list(state.positions)
        cash = list(state.cash)
        action_params = [self.quote_params[action] for action in actions]

        for player in (0, 1):
            opponent = 1 - player
            params = action_params[player]
            opponent_params = action_params[opponent]
            bid_competition = float(
                np.clip(
                    0.20 * (opponent_params["bid_edge"] - params["bid_edge"]),
                    -0.15,
                    0.15,
                )
            )
            ask_competition = float(
                np.clip(
                    0.20 * (opponent_params["ask_edge"] - params["ask_edge"]),
                    -0.15,
                    0.15,
                )
            )

            bid_fill_prob = float(
                np.clip(
                    params["bid_base"] - 0.12 * flow + bid_competition - 0.05 * abs(move),
                    0.05,
                    0.95,
                )
            )
            ask_fill_prob = float(
                np.clip(
                    params["ask_base"] + 0.12 * flow + ask_competition - 0.05 * abs(move),
                    0.05,
                    0.95,
                )
            )

            bid_draw, ask_draw = draw_row[player]
            if bid_draw < bid_fill_prob:
                positions[player] += 1
                cash[player] -= state.mid_price - params["bid_edge"]
            if ask_draw < ask_fill_prob:
                positions[player] -= 1
                cash[player] += state.mid_price + params["ask_edge"]

        next_mid_price = round(state.mid_price + move, 2)
        return tuple(positions), tuple(cash), next_mid_price, move

    @staticmethod
    def _action_token(round_index: int, player: int, action: str) -> str:
        return f"r{round_index}-p{player}-{action}"

    @staticmethod
    def _move_token(move: float) -> str:
        if move > 0.05:
            return "moveup"
        if move < -0.05:
            return "movedown"
        return "moveflat"
