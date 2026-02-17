from trading_sim import MarketMakingGame
from evaluate import exploitability
import matplotlib.pyplot as plt

mm = MarketMakingGame()
avg_util = mm.train(10000)
print(f"Average EV: {avg_util}")

strategies = {k: v.get_average_strategy() for k, v in mm.infosets.items()}
exp = exploitability(mm, strategies)
print(f"Exploitability: {exp}")

# Plot
infoset = mm.infosets['start_0']  # Sample
avg_strat = infoset.get_average_strategy()
plt.bar(mm.actions, avg_strat)
plt.title('Average Quoting Strategy')
plt.savefig('results/trading_strategy.png')
