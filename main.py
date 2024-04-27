from calendar import c
from collections import deque
import pdb

import gym

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import ipdb

from simulate import MarketSimulation

from environments.market_env import MarketEnv

# Define the neural network model
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def adjust_input_size(self, new_input_size):
        self.fc1 = nn.Linear(new_input_size, self.hidden_size)

class Export:
    def __init__(self, training_penalty, training_no_penalty):
        self.training_penalty = training_penalty
        self.training_no_penalty = training_no_penalty

        self.episode_numbers = [ training_penalty.episode_numbers_list, training_no_penalty.episode_numbers_list ]
        self.episode_rewards = [ training_penalty.episode_rewards_list, training_no_penalty.episode_rewards_list ]
        self.episode_actions = [ training_penalty.episode_actions_list, training_no_penalty.episode_actions_list ]

        self.steps_till_stop = [ training_penalty.steps_till_stop, training_no_penalty.steps_till_stop ]
        self.true_asset_price = [ training_penalty.true_asset_price, training_no_penalty.true_asset_price ]

        self.numpy_rewards = []
        self.numpy_bids = []
        self.numpy_asks = []

        for i in range(2):
            self.numpy_rewards.append([ reward.detach().numpy() for reward in self.episode_rewards[i] ])
            self.numpy_bids.append([ action[0][0].detach().numpy() for action in self.episode_actions[i] ])
            self.numpy_asks.append([ action[0][1].detach().numpy() for action in self.episode_actions[i] ])

    def plot_training(self):
        cumulative_rewards_penalty = np.cumsum(self.numpy_rewards[0])
        cumulative_steps_penalty = np.cumsum(self.steps_till_stop[0])

        cumulative_rewards_no_penalty = np.cumsum(self.numpy_rewards[1])
        cumulative_steps_no_penalty = np.cumsum(self.steps_till_stop[1])

        diff_ask_bid_penalty = np.array(self.numpy_asks[0]) - np.array(self.numpy_bids[0])
        diff_ask_bid_no_penalty = np.array(self.numpy_asks[1]) - np.array(self.numpy_bids[1])

        # set global plot font size
        plt.rcParams.update({'font.size': 14})





        # show the ask-bid spread over time, per 20 episodes. So make 1 plot and divide it into total number of episodes / 20. Each part has two box plots, one for penalty and one for no penalty
        plt.figure(figsize=(10, 4))
        num_parts = len(self.episode_numbers[0]) // 20
        # Create lists to store data for penalty and no penalty
        data_penalty = []
        data_no_penalty = []

        # Loop through each part
        for i in range(num_parts):
            # Calculate the start and end indices for this part
            start_index = i * 20
            end_index = (i + 1) * 20

            # Extract the data for this part
            diff_ask_bid_penalty_part = diff_ask_bid_penalty[start_index:end_index]
            diff_ask_bid_no_penalty_part = diff_ask_bid_no_penalty[start_index:end_index]

            # Append data to lists
            data_penalty.append(diff_ask_bid_penalty_part)
            data_no_penalty.append(diff_ask_bid_no_penalty_part)

        # Create box plots for penalty and no penalty side-by-side
        plt.boxplot(data_penalty, positions=[i*2 for i in range(num_parts)], labels=[f"{i*20+1}-{(i+1)*20}" for i in range(num_parts)], widths=0.2, boxprops=dict(color='blue'))
        plt.boxplot(data_no_penalty, positions=[i*2+0.4 for i in range(num_parts)], widths=0.2, boxprops=dict(color='red'))

        # Add labels and title
        plt.xlabel('Range of Episodes (Inclusive)')
        plt.ylabel('Ask-Bid Spread')
        plt.xticks([i*2+0.4 for i in range(num_parts)], [f"{i*20+1}-{(i+1)*20}" for i in range(num_parts)], rotation=45)

        plt.legend(['Penalty', 'No Penalty'], loc='lower right')
        plt.tight_layout()
        plt.savefig('ask_bid_spread_over_time.pdf')






        plt.figure(figsize=(10, 4))
        plt.plot(self.episode_numbers[0], self.numpy_rewards[0], color='blue', alpha=0.5, label = 'Penalty')
        plt.plot(self.episode_numbers[1], self.numpy_rewards[1], color='red', alpha=0.5, label = 'No Penalty')
        plt.xlabel('Episode Number')
        plt.ylabel('Reward')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.legend()
        plt.tight_layout()
        plt.savefig('reward.pdf')





        plt.figure(figsize=(10, 4))
        plt.plot(self.episode_numbers[0], cumulative_rewards_penalty, color='blue', alpha=0.5, label = 'Penalty')
        plt.plot(self.episode_numbers[1], cumulative_rewards_no_penalty, color='red', alpha=0.5, label = 'No Penalty')
        plt.xlabel('Episode Number')
        plt.ylabel('Cumulative Reward')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.legend()
        plt.tight_layout()
        plt.savefig('cumulative_reward.pdf')





        plt.figure(figsize=(10, 4))
        plt.plot(self.episode_numbers[0], cumulative_steps_penalty, color='blue', alpha=0.5, label = 'Penalty')
        plt.plot(self.episode_numbers[1], cumulative_steps_no_penalty, color='red', alpha=0.5, label = 'No Penalty')
        plt.xlabel('Episode Number')
        plt.ylabel('Cumulative Steps')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.legend()
        plt.tight_layout()
        plt.savefig('cumulative_steps.pdf')





        plt.show()

class Training:
    def __init__(self, policy_network, optimizer, env, gamma, max_steps, num_episodes, replay_buffer_size, only_profit=False):
        self.policy_network = policy_network
        self.optimizer = optimizer
        self.env = env
        self.gamma = gamma
        self.max_steps = max_steps
        self.num_episodes = num_episodes

        self.replay_buffer = deque(maxlen=replay_buffer_size)

        self.episode_rewards_list = []
        self.episode_numbers_list = []
        self.episode_actions_list = []
        self.steps_till_stop = []
        self.true_asset_price = []
        self.bid_prices = []
        self.ask_prices = []

        self.only_profit = only_profit

    # Function to select an action based on the policy network output
    def select_action(self, state):
        self.policy_network.adjust_input_size(state.size()[0])
        next_state = self.policy_network(state)
        next_state_copy = torch.clamp(next_state, min=0, max=1000)

        return next_state_copy

    # Function to compute the discounted rewards
    def compute_discounted_rewards(self, rewards):
        # create empty tensor
        discounted_rewards = torch.zeros(len(rewards))
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add

        return discounted_rewards

    # min loss, so we maximize profit
    def loss_function(self, revenue, expenses, inventory, current_price, cash):
        min_cash = 100
        max_inventory = 100

        # Compute profit and inventory value
        profit = revenue - expenses

        # Penalize low cash and high inventory
        cash_penalty = max(0, min_cash - cash)
        inventory_penalty = max(0, (inventory - max_inventory) * current_price)

        # Encourage profit
        profit_reward = -profit

        if self.only_profit:
            return profit_reward
        else:
            # Combine penalties and reward to form the loss function
            loss = profit_reward + cash_penalty + inventory_penalty
            return loss

    # Main loop
    def train(self):
        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            episode_rewards = []

            episode_states = []
            episode_actions = []
            
            step = 0
            while step < self.max_steps:
                step += 1
                action = self.select_action(state)
                next_state, reward, done, _, _= self.env.step(action)
                self.true_asset_price.append(self.env.market.current_price)
                self.bid_prices.append(self.env.market.current_buyer_maximums)
                self.ask_prices.append(self.env.market.current_buyer_maximums)

                episode_rewards.append(reward)
                episode_states.append(state)
                episode_actions.append(action)

                state = next_state

                if done:
                    break
            
            self.steps_till_stop.append(step)

            discounted_rewards = self.compute_discounted_rewards(episode_rewards)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std(unbiased=False) + 1e-9)

            # Update policy network
            self.optimizer.zero_grad()
            state = episode_states[-1]
            action = episode_actions[-1]
            reward = discounted_rewards[-1]

            self.policy_network.adjust_input_size(state.size()[0])
            output_tensor = self.policy_network(state)

            inventory = state[-2]
            current_price = env.market.current_price
            expenses = env.market.get_expenses(action[0])
            revenue = env.market.get_revenue(action[1])

            loss = self.loss_function(revenue, expenses, inventory, current_price, state[-1])
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
            self.optimizer.step()

            # store episode rewards and episode number
            self.episode_rewards_list.append(sum(episode_rewards))
            self.episode_numbers_list.append(episode + 1)
            self.episode_actions_list.append(episode_actions)

            # Print episode information
            total_reward = sum(episode_rewards)

            print(f"Episode {episode + 1}, Total Reward: {total_reward}, Steps: {step}")

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
hidden_size = 20
num_episodes = 400
# num_episodes = 80
max_steps = 200
# max_steps = 10
replay_buffer_size = 1000

market = MarketSimulation()

# Initialize environment and policy network

history = {
    'bid': [],
    'ask': [],
    'profit': [],
    'buyer': [],
    'seller': [],
    'inventory': [],
    'cash': []
}

env = gym.make('MarketEnv', market=market, history=history, inventory=0, cash=1000)
input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]
policy_network = PolicyNetwork(input_size, hidden_size, output_size)
optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)
torch.autograd.set_detect_anomaly(True)

training_penalty = Training(policy_network, optimizer, env, gamma, max_steps, num_episodes, replay_buffer_size)
training_penalty.train()

training_no_penalty = Training(policy_network, optimizer, env, gamma, max_steps, num_episodes, replay_buffer_size, only_profit=True)
training_no_penalty.train()

env.close()

export = Export(training_penalty, training_no_penalty)
export.plot_training()