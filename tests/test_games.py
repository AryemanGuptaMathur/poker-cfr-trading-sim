import math

import numpy as np

from poker_cfr import PokerGame, PokerState
from trading_sim import MarketMakingGame, MarketScenario, TradingState


def test_poker_infosets_use_underscore_delimited_history():
    game = PokerGame(seed=0)
    state = PokerState(cards=(3, 1), history=("check", "bet"), player_to_act=0)

    assert game.info_key(state, player=0) == "K_check_bet"


def test_poker_fold_utility_matches_contributions():
    game = PokerGame(seed=0)
    state = PokerState(cards=(3, 1), history=("bet", "fold"), player_to_act=0)

    assert game.terminal_utility(state, player=0) == 1.0
    assert game.terminal_utility(state, player=1) == -1.0


def test_trading_terminal_utility_includes_inventory_penalty():
    game = MarketMakingGame(seed=0, horizon=1, sigma=0.5, num_scenarios=4)
    scenario = MarketScenario(
        order_flow=(0,),
        price_moves=(0.0,),
        fill_draws=(((0.5, 0.5), (0.5, 0.5)),),
    )
    state = TradingState(
        scenario=scenario,
        history=("r0-p0-narrow", "r0-p1-wide", "moveflat"),
        round_index=1,
        player_to_act=0,
        pending_actions=(None, None),
        positions=(4, 0),
        cash=(1.0, 0.0),
        mid_price=100.0,
        last_move=0.0,
    )

    expected_player_0 = 1.0 + 4 * 100.0 - 0.5 * math.sqrt(4)
    expected_player_1 = 0.0
    expected_relative = expected_player_0 - expected_player_1

    assert game.terminal_utility(state, player=0) == expected_relative
    assert game.terminal_utility(state, player=1) == -expected_relative


def test_play_with_best_response_reaches_terminal_state():
    game = PokerGame(seed=1)
    strategy_profile = {
        "J_start": np.array([0.5, 0.5]),
        "Q_start": np.array([0.5, 0.5]),
        "K_start": np.array([0.5, 0.5]),
    }

    terminal_state = game.play_with_br(strategy_profile, br_player=0, seed=7)

    assert game.is_terminal(terminal_state)
