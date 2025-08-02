import numpy as np


class LinearSkew:
    def __init__(self, a1=0.1, a2=0.0, a3=0.0):
        self.alpha = a1
        self.beta = a2

    def __call__(self, state, ttc=0, eps=1e-3):
        # Define a sigmoid-based skew function
        position = state.get("position", 0)
        ttc = state.get("time_remaining", ttc)

        # Sigmoid function for smooth transition
        skew = position * self.alpha / ((ttc + eps) ** self.beta)
        skew = np.clip(skew, -1, 1)  # Ensure skew is within [-1, 1]
        return 1 + skew, 1 - skew
