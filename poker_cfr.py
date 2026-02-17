import numpy as np
from utils import Node, chance_sample

class LeducPoker:
    """Simplified Leduc Poker environment."""
    def __init__(self):
        self.deck = [1,1,1,2,2,2]  # Ranks 1 (Jack), 2 (Queen), 3 suits each
        self.actions = ['fold', 'check', 'bet']  # Simplified
        self.infosets = defaultdict(lambda: Node(self.actions))

    def cfr(self, history, player, opp_prob=1.0, my_prob=1.0):
        """Recursive CFR traversal."""
        if self.is_terminal(history):
            return self.utility(history, player)
        
        infoset = self.get_infoset(history, player)
        strategy = infoset.get_strategy(my_prob)
        action_values = np.zeros(len(self.actions))
        
        for i, action in enumerate(self.actions):
            next_history = history + action
            if player == 0:  # Current player
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
            np.random.shuffle(self.deck)
            history = ''  # Start with deal
            util += self.cfr(history, 0)  # Player 0 starts
        return util / iterations

    # Helpers
    def get_infoset(self, history, player):
        # Infoset: hand + history (hide opponent's hand)
        hand = 'J' if self.deck[player] == 1 else 'Q'  # Example
        return self.infosets[hand + '_' + self.public_history(history)]

    def public_history(self, history):
        return ''.join([a[0] for a in history.split('_') if a])  # Abbrev

    def is_terminal(self, history):
        return 'fold' in history or len(history.split('_')) >= 4  # Max depth

    def utility(self, history, player):
        if 'fold' in history:
            return 1 if history[-4:] == 'fold' else -1  # Winner gets pot
        # Showdown: compare hands (simplified)
        p0_hand, p1_hand = self.deck[0], self.deck[1]
        return 1 if p0_hand > p1_hand else -1 if player == 0 else 1

# Example usage in run_poker.py
