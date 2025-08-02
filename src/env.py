import numpy as np
import pandas as pd


class MarketMakingEnv:
    def __init__(
        self,
        liquidity_sensitivity,
        standard_spread,
        brokerage,
        max_position,
        mid_price=100.0,
        price_volatility=0.1,
        demand_scale=10,
        start_time=360,  # 6am in minutes
        end_time=960,  # 4pm in minutes
        time_step=10,  # in minutes
    ):
        self.liquidity_sensitivity = liquidity_sensitivity
        self.standard_spread = standard_spread
        self.brokerage = brokerage
        self.max_position = max_position
        self.start_time = start_time
        self.end_time = end_time
        self.time_step = time_step
        self.current_time = self.start_time
        self.position = 0
        self.done = False
        self.mid_price = mid_price
        self.price_volatility = price_volatility
        self.cash = 0.0
        self.demand_scale = demand_scale

    def reset(self):
        self.current_time = self.start_time
        self.position = 0
        self.done = False
        self.mid_price = 100.0  # Reset mid-price to initial value
        self.cash = 0.0
        return self._get_state()

    def fill_probability(self, skew):
        # use sigmoid function for fill probability
        return np.exp(-skew * self.liquidity_sensitivity)

    def step(self, action):
        if self.done:
            raise ValueError("Episode has ended. Please reset the environment.")

        # Update mid-price with random walk
        delta_mid = np.random.normal(loc=0, scale=self.price_volatility)
        self.mid_price += delta_mid
        mark_to_parket_pnl = self.position * delta_mid

        bid_spread, offer_spread = action
        bid_spread = max(bid_spread, 0)
        offer_spread = max(offer_spread, 0)

        # Calculate fill probabilities
        bid_fill_prob = self.fill_probability(bid_spread)
        offer_fill_prob = self.fill_probability(offer_spread)

        # Simulate random bid and offer demand
        bid_demand = np.random.exponential(scale=self.demand_scale)
        offer_demand = np.random.exponential(scale=self.demand_scale)

        # Simulate fills
        bid_fill = np.random.binomial(bid_demand, bid_fill_prob)
        offer_fill = np.random.binomial(offer_demand, offer_fill_prob)

        # Update position and cash
        self.position += bid_fill
        self.cash -= bid_fill * (self.mid_price - bid_spread)
        self.position -= offer_fill
        self.cash += offer_fill * (self.mid_price + offer_spread)

        inception_pnl = (
            bid_fill * bid_spread + offer_fill * offer_spread
        ) * self.standard_spread

        # Advance time
        self.current_time += self.time_step

        if self.current_time >= self.end_time:
            self.done = True

        liquidation_penalty = 0
        if self.done:
            if self.position < 0:
                liquidation_penalty = (self.standard_spread + self.brokerage) * abs(
                    self.position
                )
                self.cash += self.position * self.mid_price - liquidation_penalty
                self.position = 0  # Reset position to zero after liquidation

        reward = inception_pnl + mark_to_parket_pnl - liquidation_penalty

        return self._get_state(), reward, self.done

    def _get_state(self):
        return {
            "time": self.current_time,
            "position": self.position,
            "time_remaining": self.end_time - self.current_time,
            "cash": self.cash,
            "mid_price": self.mid_price,
            "pnl": self.cash + self.position * self.mid_price,
        }

    def render(self):
        print(f"Time: {self.current_time // 60}:{self.current_time % 60:02d}")
        print(f"Position: {self.position}")
        print(f"Mid Price: {self.mid_price:.2f}")
        print(f"cash: {self.cash:.2f}")
        print(f"Time Remaining: {self.end_time - self.current_time} minutes")


if __name__ == "__main__":
    env = MarketMakingEnv(
        liquidity_sensitivity=0.1,
        standard_spread=0.01,
        brokerage=0.001,
        max_position=100,
    )
    state = env.reset()
    print("Initial State:", state)
    action = (0.02, 0.02)  # Example action
    done = False
    history = []

    while not done:
        history.append(state)
        next_state, reward, done, _ = env.step(action)
        print("Next State:", next_state)
        print("Reward:", reward)
        state = next_state
        env.render()
    history.append(state)
    history = pd.DataFrame(history)

    # plot history
    import matplotlib.pyplot as plt

    plt.plot(history["time"], history["pnl"])
    plt.xlabel("Time")
    plt.ylabel("PnL")
    plt.title("PnL Over Time")
    plt.savefig("pnl_over_time.png")
    # Save history to CSV
    history.to_csv("history.csv", index=False)
    print("History saved to history.csv")
