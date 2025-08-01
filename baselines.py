import numpy as np


class SigmoidAgent:
    def __init__(self, alpha=0.1, beta=0.0, min_spread=0.0, max_spread=2.0):
        self.alpha = alpha
        self.beta = beta
        self.min_spread = min_spread
        self.max_spread = max_spread

    def __call__(self, state, ttc=0, eps=1e-3):
        # Define a sigmoid-based skew function
        position = state.get("position", 0)
        ttc = state.get("time_remaining", ttc)

        # Sigmoid function for smooth transition
        scale_factor = position * self.alpha / ((ttc + eps) ** self.beta)
        bid_skew = self.min_spread + (self.max_spread - self.min_spread) / (
            1 + np.exp(scale_factor)
        )
        offer_skew = self.min_spread + (self.max_spread - self.min_spread) / (
            1 + np.exp(- scale_factor)
        )
        return bid_skew, offer_skew

class LinearAgent:
    def __init__(self, alpha=0.1, beta=0.0, min_spread=0.0, max_spread=2.0):
        self.alpha = alpha
        self.beta = beta
        self.min_spread = min_spread
        self.max_spread = max_spread

    def __call__(self, state, ttc=0, eps=1e-3):
        # Define a sigmoid-based skew function
        position = state.get("position", 0)
        ttc = state.get("time_remaining", ttc)

        # Sigmoid function for smooth transition
        scale_factor = position * self.alpha + self.beta
        bid_skew = - position * self.alpha + self.beta
        bid_skew = min(max(bid_skew, self.min_spread), self.max_spread)

        offer_skew = position * self.alpha + self.beta
        offer_skew = min(max(offer_skew, self.min_spread), self.max_spread)
        return bid_skew, offer_skew

class TanhAgent:
    def __init__(self, alpha=0.1, beta=0.0, max_skew=1.0):
        self.alpha = alpha
        self.beta = beta
        self.max_skew = max_skew

    def __call__(self, state, ttc=0, eps=1e-3):
        # Define a tanh-based skew function
        position = state.get("position", 0)
        ttc = state.get("time_remaining", ttc)

        scale_factor = position * self.alpha / ((ttc + eps) ** self.beta)

        skew = self.max_skew * np.tanh( - scale_factor)
        
        return 1-skew, 1+skew
    

class ExponentialAgent:
    def __init__(self, alpha=0.1, beta=0.0, min_spread=0.0, max_spread=2.0):
        self.alpha = alpha
        self.beta = beta
        self.min_spread = min_spread
        self.max_spread = max_spread

    def __call__(self, state, ttc=0, eps=1e-3):
        position = state.get("position", 0)
        ttc = state.get("time_remaining", ttc)

        bid_skew = self.min_spread + (self.max_spread - self.min_spread) * np.exp(
            -self.alpha * position / (ttc + eps) ** self.beta
        )
        offer_skew = self.min_spread + (self.max_spread - self.min_spread) * np.exp(
            self.alpha * position / (ttc + eps) ** self.beta
        )
        return bid_skew, offer_skew