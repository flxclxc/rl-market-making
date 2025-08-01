from collections import deque
import numpy as np
from torch import nn, optim
import torch
import yaml

from env import MarketMakingEnv

# Policy network (Actor)
class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Value network (Critic)
class ValueNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

if __name__ == '__main__':
    state_dim = 2  # Example state dimension
    action_dim = 2  # Example action dimension
    lr = 0.001  # Learning rate
    gamma = 1.0  # Discount factor

    # Instantiate components
    actor = PolicyNet(state_dim, action_dim)
    critic = ValueNet(state_dim)
    actor_optim = optim.Adam(actor.parameters(), lr=lr)
    critic_optim = optim.Adam(critic.parameters(), lr=lr)

    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
        
    running_rewards = deque(maxlen=100)  # Store last 100 rewards

    for _ in range(1000):  # Example number of steps        
        log_probs = []
        rewards = []
        states = []

        env = MarketMakingEnv(**config['env'])  
        state = env.reset()

        while not env.done:
            state_tensor = torch.tensor([state['position'], state['time_remaining']], dtype=torch.float32)
            states.append(state_tensor)
            action = actor(state_tensor)
            dist = torch.distributions.Normal(action, 0.1)
            sampled_action = dist.rsample()
            log_prob = dist.log_prob(sampled_action).sum()
                
            value = critic(state_tensor)
            next_state, reward, done = env.step(sampled_action.detach().numpy())

            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state
        
        running_rewards.append(np.sum(rewards))
        print(np.mean(running_rewards))
        state_tensor = torch.tensor([state['position'], state['time_remaining']], dtype=torch.float32)
        states.append(state_tensor)

        # --- In your training loop, after collecting an episode:
        states = torch.stack(states)
        values = critic(states)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        td_target = rewards + gamma * values[1:]  # Compute returns
        advantage = td_target - values[:-1]
        log_probs = torch.stack(log_probs)

        actor_loss = (-log_probs * advantage.detach()).sum()
        critic_loss = nn.MSELoss()(values[:-1], td_target.detach())
    
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        # Update critic
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()
        