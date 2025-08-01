# RL Market Making

A project exploring reinforcement learning for market making in FX markets. Agent sets bid and offer spread as a function of a fixed core spread. Constrained problem with forced liquidation at end of session.

Environment Model:

- Buy/Sell demand ~ Exp(demand_scale)
- p(FILL) = exp(-liquidity_sensitivity * skew_spread)
- Brownian motion mid price process
