import numpy as np
from utils import Node, chance_sample

class MarketMakingGame:
    """Simplified market-making as a game."""
    def __init__(self):
        self.actions = ['narrow_spread', 'wide_spread', 'skew_bid', 'skew_ask']  # Quoting actions
        self.infosets = defaultdict(lambda: Node(self.actions))
        self.market_states = [0, 1, 2]  # 0: neutral, 1: buy pressure, 2: sell (hidden to agent partially)

    def cfr(self, history, player, opp_prob=1.0, my_prob=1.0):
        if self.is_terminal(history):
            return self.utility(history, player)
        
        infoset = self.get_infoset(history, player)
        strategy = infoset.get_strategy(my_prob)
        action_values = np.zeros(len(self.actions))
        
        for i, action in enumerate(self.actions):
            next_history = history + '_' + action
            # Chance: Market move
            market_move = chance_sample(self.market_states)
            next_history += f'_{market_move}'
            if player == 0:
                action_values[i] = -self.cfr(next_history, player, opp_prob, my_prob * strategy[i])
            else:
                action_values[i] = -self.cfr(next_history, player, opp_prob * strategy[i], my_prob)
        
        node_value = np.dot(strategy, action_values)
        regrets = action_values - node_value
        infoset.regret_sum += opp_prob * regrets
        
        return node_value

    def train(self, iterations=10000):
        util = 0
        for i in range(iterations):
            history = 'start'
            util += self.cfr(history, 0)
        return util / iterations

    # Helpers
    def get_infoset(self, history, player):
        # Partial obs: Last action + visible market (hide full state)
        visible = '_'.join([part for part in history.split('_') if int(part) % 2 == 0])  # Example partial
        return self.infosets[visible]

    def is_terminal(self, history):
        return len(history.split('_')) >= 5  # Sim rounds

    def utility(self, history, player):
        # EV: Profit from fills minus inventory risk
        actions = history.split('_')[1::2]
        market_moves = [int(m) for m in history.split('_')[2::2]]
        profit = sum(1 if a == 'narrow_spread' and m == 0 else -1 if a == 'wide_spread' and m != 0 else 0 for a, m in zip(actions, market_moves))
        return profit if player == 0 else -profit

# Adapt poker bot: Use similar training loop in run_trading.py
