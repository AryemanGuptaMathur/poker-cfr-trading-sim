from poker_cfr import LeducPoker
import matplotlib.pyplot as plt

poker = LeducPoker()
avg_util = poker.train(10000)
print(f"Average utility: {avg_util}")

# Plot strategies for a sample infoset
infoset = poker.infosets['J_check']
avg_strat = infoset.get_average_strategy()
plt.bar(poker.actions, avg_strat)
plt.title('Average Strategy for Infoset J_check')
plt.savefig('results/poker_strategy.png')
