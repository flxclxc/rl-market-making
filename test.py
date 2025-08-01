from matplotlib import pyplot as plt
import pandas as pd
from env import MarketMakingEnv
import yaml

from skew_agent import SkewAgent
from utils import play_episode, plot_results
import numpy as np

if __name__ == "__main__":
    # Load configuration from config.yml
    with open("env_config.yml", "r") as file:
        config = yaml.safe_load(file)

    # Pass configuration to the environment if needed
    histories = []
    for i in range(1000):
        env = MarketMakingEnv(**config)
        skew_agent = SkewAgent(
            alpha=1.0, beta=0.0, min_spread=0.5, max_spread=1.5
        )

        history = play_episode(env, skew_agent)
        history['seed'] = i
        histories.append(history)

    plot_results(history)
    histories = pd.concat(histories)
    print('EOD PNL:')
    eod_pnl = histories.groupby('seed')['pnl'].last()
    print(eod_pnl.describe())
    eod_pnl.hist()
    plt.savefig('eod.png')