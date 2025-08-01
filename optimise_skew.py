import argparse
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import yaml
from utils import play_episode, plot_results
from env import MarketMakingEnv
from baselines import LinearAgent, SigmoidAgent, TanhAgent
from hyperopt.early_stop import no_progress_loss

# Define the CARA utility function
def utility_func(profit, risk_aversion):
    return np.mean(profit) - risk_aversion * np.std(profit)

# Objective function for hyperopt
def objective(params):
    print(params)
    
    # Initialize the environment and agent
    agent = TanhAgent(**params)
    
    # Run the agent in the environment
    pnls = []

    for _ in range(1000):  # Number of episodes
        history = play_episode(env, agent)
        pnls.append(history['pnl'].iloc[-1])

    # Compute CARA utility
    risk_aversion = 0.5  # Adjust risk aversion as needed
    utility = utility_func(np.array(pnls), risk_aversion)
    
    # Hyperopt minimizes the objective, so return the negative utility
    return {'loss': - utility, 'status': STATUS_OK}

# # Define the search space
# space = {
#     'alpha': hp.loguniform('alpha', -3, 3),
#     'beta' : hp.uniform('beta', 0.0, 2.0),
#     'min_spread': hp.uniform('min_spread', 0.00, 0.5),
#     'max_spread': hp.uniform('max_spread', 1.0, 2.0),
# }

space = {
    'alpha': hp.loguniform('alpha', -3, 3),
    'beta' : hp.uniform('beta', 0.0, 2.0),
    'max_skew': hp.uniform('max_skew', 0.0, 1.0),
}

# # Define the search space
# space = {
#     'alpha': hp.loguniform('alpha', -5, 5),
#     'beta' : hp.uniform('beta', -1, 1.0),
#     'min_spread': hp.uniform('min_spread', 0.00, 0.5),
#     'max_spread': hp.uniform('max_spread', 1.0, 2.0),
# }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for Skew Agent")
    parser.add_argument('--tune', action='store_true')
    args = parser.parse_args()

    # Load configuration from config.yml
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    # Initialize the environment with the loaded configuration
    env = MarketMakingEnv(**config['env'])
    skew_agent = TanhAgent()

    if args.tune:
        # Run Bayesian optimization
        trials = Trials()

        # Define early stopping criterion
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=500,  # Number of evaluations
            trials=trials,
            early_stop_fn=no_progress_loss(50)  # Pass the early stopping function
        )
        for k,v in best.items():
            best[k] = float(v) if isinstance(v, np.float64) else v
        
        print("Best parameters:", best)
        
        skew_agent = TanhAgent(**best)

        config['tanh_agent'] = best
        
        with open("config.yml", "w") as file:
            yaml.dump(config, file)

    histories = []

    print("Running episodes with best parameters...") 
    for i in range(10000):
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
