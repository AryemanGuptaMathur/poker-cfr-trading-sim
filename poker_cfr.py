from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from abstract_game import AbstractGame


CARD_LABELS = {1: "J", 2: "Q", 3: "K"}
ACTION_VOCAB = ("<pad>", "check", "bet", "call", "fold")


@dataclass(frozen=True)
class PokerState:
    cards: tuple[int, int]
    history: tuple[str, ...]
    player_to_act: int


class PokerGame(AbstractGame):
    """Toy poker game with Kuhn-style betting and CFR training."""

    def __init__(self, seed: int = 0, variant: str = "vanilla") -> None:
        variant_suffix = "CFR+" if variant == "cfr+" else "Vanilla CFR"
        super().__init__(
            name=f"Toy Poker {variant_suffix}",
            all_actions=["check", "bet", "call", "fold"],
            seed=seed,
            variant=variant,
        )
        self.deck = (1, 2, 3)
        self.max_history_tokens = 3
        self.encoding_size = 2 + 3 + self.max_history_tokens * len(ACTION_VOCAB)
        self.action_index = {token: index for index, token in enumerate(ACTION_VOCAB)}

    def sample_initial_state(self, rng: np.random.Generator) -> PokerState:
        cards = tuple(int(card) for card in rng.choice(self.deck, size=2, replace=False))
        return PokerState(cards=cards, history=tuple(), player_to_act=0)

    def enumerate_initial_states(self) -> list[tuple[PokerState, float]]:
        states = []
        probability = 1.0 / 6.0
        for card_0 in self.deck:
            for card_1 in self.deck:
                if card_0 == card_1:
                    continue
                states.append(
                    (
                        PokerState(cards=(card_0, card_1), history=tuple(), player_to_act=0),
                        probability,
                    )
                )
        return states

    def is_terminal(self, state: PokerState) -> bool:
        return state.history in {
            ("check", "check"),
            ("bet", "call"),
            ("bet", "fold"),
            ("check", "bet", "call"),
            ("check", "bet", "fold"),
        }

    def current_player(self, state: PokerState) -> int:
        return state.player_to_act

    def legal_actions(self, state: PokerState) -> list[str]:
        if not state.history:
            return ["check", "bet"]
        if state.history == ("check",):
            return ["check", "bet"]
        if state.history in {("bet",), ("check", "bet")}:
            return ["call", "fold"]
        return []

    def next_state(self, state: PokerState, action: str) -> PokerState:
        return PokerState(
            cards=state.cards,
            history=state.history + (action,),
            player_to_act=1 - state.player_to_act,
        )

    def info_key(self, state: PokerState, player: int) -> str:
        return f"{CARD_LABELS[state.cards[player]]}_{self.history_to_string(state.history)}"

    def terminal_utility(self, state: PokerState, player: int) -> float:
        contributions = [1, 1]
        acting_player = 0

        for action in state.history:
            if action in {"bet", "call"}:
                contributions[acting_player] += 1
            if action == "fold":
                winner = 1 - acting_player
                return float(contributions[1 - player] if winner == player else -contributions[player])
            acting_player = 1 - acting_player

        winner = 0 if state.cards[0] > state.cards[1] else 1
        return float(contributions[1 - player] if winner == player else -contributions[player])

    def encode_state(self, state: PokerState, player: int) -> np.ndarray:
        vector = np.zeros(self.encoding_size, dtype=np.float32)

        vector[player] = 1.0
        vector[2 + (state.cards[player] - 1)] = 1.0

        offset = 5
        for index in range(self.max_history_tokens):
            token = state.history[index] if index < len(state.history) else "<pad>"
            token_index = self.action_index[token]
            vector[offset + index * len(ACTION_VOCAB) + token_index] = 1.0

        return vector


LeducPoker = PokerGame
