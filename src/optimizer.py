import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, tpe
from hyperopt.early_stop import no_progress_loss

from utils import play_episode


class SkewOptimizer:
    def __init__(self, n_episodes=1000, risk_aversion=0.5):
        self.n_episodes = n_episodes
        self.risk_aversion = risk_aversion

    def utility_func(self, profit):
        return np.mean(profit) - self.risk_aversion * np.std(profit)

    def objective(self, params):
        print(params)

        # Initialize the environment and agent
        skew = self.skew_func(**params)
        pnls = []

        for _ in range(self.n_episodes):
            history = play_episode(self.env, skew)
            pnls.append(history["pnl"].iloc[-1])

        # Compute CARA utility
        utility = self.utility_func(np.array(pnls))

        # Hyperopt minimizes the objective, so return the negative utility
        return {"loss": -utility, "status": STATUS_OK}

    def __call__(self, env, skew_func, param_space, n_trials=500, patience=50):
        trials = Trials()
        self.skew_func = skew_func
        self.env = env

        best = fmin(
            fn=self.objective,
            space=param_space,
            algo=tpe.suggest,
            max_evals=n_trials,
            trials=trials,
            early_stop_fn=no_progress_loss(patience),
        )

        for k, v in best.items():
            best[k] = float(v) if isinstance(v, np.float64) else v

        print("Best parameters:", best)
        return best
