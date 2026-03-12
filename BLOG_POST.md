# Draft Blog Post

## Bridging Game Theory and Market Making with CFR

I built this project to connect two interests that usually live in separate folders on a resume: algorithmic game theory and quantitative trading intuition.

The core idea is Counterfactual Regret Minimization, a classic algorithm for learning strategies in imperfect-information games. I started with poker because it is the cleanest way to test whether CFR logic is actually correct. If private information, alternating turns, or regret updates are wrong there, any finance-flavored extension is just window dressing.

Once the poker side worked, I extended the same framework into a small market-making game. The trading environment is still intentionally toy-sized, but it is no longer pure random chance. Each training iteration samples a synthetic order-flow path, players choose quoting actions like `narrow`, `wide`, or skewed quotes, and the game settles fills, mark-to-market P&L, and inventory penalties.

The most important lesson from the build was not “AI beats markets.” It was the opposite: getting the basics right matters more than adding flashy ideas. The biggest improvements came from fixing the recursion, adding both-player traversals, evaluating exploitability, and writing tests around edge cases.

I also compared plain tabular CFR against CFR+ and a small Deep CFR style approximation that learns regret targets with a neural network. The deep version is noisier, but it makes the project a useful sandbox for discussing when function approximation helps and when tabular methods are still the better engineering choice.

This project sits in the middle of CS and finance in a way I like: math-heavy enough to be interesting, simple enough to explain honestly, and small enough that every design decision is visible.

Replace this placeholder with your own Medium link:

- `[Insert your Medium article on investing or risk here](https://medium.com/)`
