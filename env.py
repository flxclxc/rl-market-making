import numpy as np


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
        return np.exp(- self.liquidity_sensitivity * skew)

    def step(self, action):
        if self.done:
            raise ValueError(
                "Episode has ended. Please reset the environment."
            )

        bid_skew, offer_skew = action
        bid_spread = self.standard_spread * (1 + bid_skew)
        offer_spread = self.standard_spread * (1 + offer_skew)

        # Calculate fill probabilities
        bid_fill_prob = self.fill_probability(bid_skew)
        offer_fill_prob = self.fill_probability(offer_skew)

        # Simulate random bid and offer demand
        bid_demand = np.random.exponential(scale=self.demand_scale)
        offer_demand = np.random.exponential(scale=self.demand_scale)

        # Simulate fills
        bid_fill = np.random.binomial(bid_demand, bid_fill_prob)
        offer_fill = np.random.binomial(offer_demand, offer_fill_prob)

        # Update position and cash
        bid_price = self.mid_price + bid_spread * self.standard_spread
        offer_price = self.mid_price - offer_spread * self.standard_spread

        self.position -= bid_fill
        self.position += offer_fill

        self.cash += bid_fill * bid_price
        self.cash -= offer_fill * offer_price

        # Advance time
        self.current_time += self.time_step

        # Update mid-price with random walk
        self.mid_price += np.random.normal(loc=0, scale=self.price_volatility)

        reward = 0

        # Check if episode ends
        if self.current_time >= self.end_time:
            self.done = True
            mkt_bid = self.mid_price + self.standard_spread
            mkt_offer = self.mid_price - self.standard_spread

            # Liquidate remaining position
            if self.position < 0:
                self.cash -= (mkt_bid + self.brokerage) * abs(self.position)
            else:
                self.cash += (
                    mkt_offer - self.brokerage
                ) * abs(self.position)

            self.position = 0

        return self._get_state(), reward, self.done, {}

    def _get_state(self):
        return {
            "time": self.current_time,
            "position": self.position,
            "time_remaining": self.end_time - self.current_time,
            "mid_price": self.mid_price,
            "cash": self.cash,
            "pnl": self.cash + self.position * self.mid_price,
        }

    def render(self):
        print(f"Time: {self.current_time // 60}:{self.current_time % 60:02d}")
        print(f"Position: {self.position}")
        print(f"Mid Price: {self.mid_price:.2f}")
        print(f"cash: {self.cash:.2f}")
        print(f"Time Remaining: {self.end_time - self.current_time} minutes")
