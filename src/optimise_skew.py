import argparse

import pandas as pd
import yaml
from hyperopt import hp
from matplotlib import pyplot as plt

from baselines import LinearSkew
from env import MarketMakingEnv
from optimizer import SkewOptimizer
from utils import play_episode, plot_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter Optimization for Skew Agent"
    )
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()

    # Load configuration from config.yml
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    # Initialize the environment with the loaded configuration
    env = MarketMakingEnv(**config["env"])
    skew_func = LinearSkew

    param_space = {"a1": hp.loguniform("a1", -3, 3)}

    if args.tune:
        optimizer = SkewOptimizer(n_episodes=500, risk_aversion=1)
        best = optimizer(env, skew_func, param_space, n_trials=500, patience=50)

        print("Best parameters:", best)

        config["linear_skew"] = best

        with open("config.yml", "w") as file:
            yaml.dump(config, file)

    skew = skew_func(**config['linear_skew'])

    histories = []
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    print("Running episodes with best parameters...")
    for i in range(1000):
        history = play_episode(env, skew)
        history["seed"] = i
        ax[0].plot(history["time"], history["pnl"], alpha=0.1, color="blue")
        histories.append(history)

    plot_results(history)

    histories = pd.concat(histories)
    print("EOD PNL:")
    eod_pnl = histories.groupby("seed")["pnl"].last()
    print(eod_pnl.describe())
    # histogram of eop pnl
    ax[1].hist(eod_pnl, bins=50, color="blue", alpha=0.7)
    ax[1].set_title("End of Day PnL Distribution")
    ax[1].set_xlabel("PnL")
    ax[1].set_ylabel("Frequency")
    ax[0].set_title("PnL Over Time")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("PnL")
    plt.tight_layout()
    plt.savefig("outputs/results.png")
