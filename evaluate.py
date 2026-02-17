from poker_cfr import LeducPoker
from trading_sim import MarketMakingGame
import numpy as np

def best_response_value(game, strategy_profile, player):
    """Compute best response EV."""
    # Traverse tree with fixed opponent strategy, maximize for player
    # Simplified: Simulate 1000 games
    utils = []
    for _ in range(1000):
        history = ''
        util = game.utility(game.play_with_br(history, strategy_profile), player)
        utils.append(util)
    return np.mean(utils)

def exploitability(game, strategies):
    br0 = best_response_value(game, strategies, 0)
    br1 = best_response_value(game, strategies, 1)
    return (br0 + br1) / 2

# EV: Average utility over sims

# Example
poker = LeducPoker()
strategies = {k: v.get_average_strategy() for k, v in poker.infosets.items()}
exp = exploitability(poker, strategies)
print(f"Exploitability: {exp}")
