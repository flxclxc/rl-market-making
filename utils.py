from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def plot_results(history):
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1)
    plt.plot(history["time"], history["position"])
    plt.title("Inventory ")
    plt.xlabel("Time Steps")
    plt.ylabel("Inventory")

    plt.subplot(3, 2, 6)
    plt.plot(history["time"], history["pnl"], color="green")
    plt.title("PnL ")
    plt.xlabel("Time Steps")
    plt.ylabel("PnL")

    plt.subplot(3, 2, 2)
    plt.plot(history["time"], history["cash"], color="blue")
    plt.title("Cash ")
    plt.xlabel("Time Steps")
    plt.ylabel("Cash")

    plt.subplot(3, 2, 3)
    plt.plot(
        history["time"],
        history["bid_spread"],
        color="orange",
    )
    plt.title("Bid Spread ")
    plt.xlabel("Time Steps")
    plt.ylabel("Bid Spread")

    plt.subplot(3, 2, 4)
    plt.plot(
        history["time"],
        history["offer_spread"],
        color="red",
    )
    plt.title("Offer Spread ")
    plt.xlabel("Time Steps")
    plt.ylabel("Offer Spread")

    plt.subplot(3, 2, 5)
    plt.plot(
        history["time"], history["mid_price"], label="Mid Price", color="blue"
    )
    plt.title("Mid Price ")
    plt.xlabel("Time Steps")
    plt.ylabel("Mid Price")

    plt.tight_layout()
    plt.savefig("results_plot.png")  # Save the figure to a file
    plt.close()  # Close the figure to free up memory


def play_episode(env, skew_func):
    state = env.reset()
    done = False

    states = []

    while not done:
        # Replace with your policy or action logic
        action = skew_func(state)
        next_state, reward, done, info = env.step(action)

        # Collect data for plotting
        state["bid_spread"] = action[0]
        state["offer_spread"] = action[1]
        states.append(state)

        state = next_state

    # Collect data for plotting
    state["bid_spread"] = action[0]
    state["offer_spread"] = action[1]
    states.append(state)

    return pd.DataFrame(states)
