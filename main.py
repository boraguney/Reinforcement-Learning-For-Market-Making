from collections import deque
import random

import gym

from sympy import plot
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
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
    def __init__(self, episode_numbers, episode_rewards, steps_till_stop, true_asset_price, bid_prices, ask_prices):
        self.episode_numbers = episode_numbers
        self.episode_rewards = episode_rewards
        self.steps_till_stop = steps_till_stop
        self.true_asset_price = true_asset_price
        self.bid_prices = bid_prices
        self.ask_prices = ask_prices

        self.numpy_rewards = np.array([reward.detach().numpy() for reward in self.episode_rewards])

    def statistics(self):
        # total rewards

        # average rewards per episode

        # ecdf of rewards over time

        # average steps till stop

        # ecdf of steps till stop

        # standard deviation of rewards

        # standard deviation of steps till stop

        # frequency of profitable trades versus losing trades, and the magnitudes of each

        # largest peak-to-trough decline in cumulative reward for each model

        # Bid-Ask Spreads

        # Cash Utilization

        # Inventory Utilization

        pass

    def plot_training(self):
        cumulative_rewards = np.cumsum(self.numpy_rewards)

        plt.figure()
        plt.plot(self.episode_numbers, self.numpy_rewards, color='blue')
        plt.xlabel('Episode Number')
        plt.ylabel('Reward per Episode')
        plt.title('Training Progress')
        plt.savefig('training_progress.pdf')

        plt.figure()
        plt.plot(self.episode_numbers, self.steps_till_stop, color='red')
        plt.xlabel('Episode Number')
        plt.ylabel('Steps Till Stop')
        plt.title('Training Progress')
        plt.savefig('steps_till_stop.pdf')

        plt.figure()
        plt.plot(self.episode_numbers, cumulative_rewards, color='green')
        plt.xlabel('Episode Number')
        plt.ylabel('Cumulative Reward')
        plt.title('Training Progress')
        plt.savefig('cumulative_reward.pdf')

        # plt.figure()
        # plt.plot(self.true_asset_price, color='black', linewidth=2)
        # for i, (bid_prices, ask_prices) in enumerate(zip(self.bid_prices, self.ask_prices)):
        #     x_values = [i] * len(bid_prices)
        #     plt.plot(x_values, bid_prices, 'ro', alpha=0.5, markersize=2.5)
        #     plt.plot(x_values, ask_prices, 'bo', alpha=0.5, markersize=2.5)
        # plt.xlabel('Time')
        # plt.ylabel('Asset Price')
        # plt.title('Asset Price')
        # plt.legend(['Asset Price', 'Bid Prices', 'Ask Prices'])
        # plt.savefig('asset_price.png')

        plt.show()

class Training:
    def __init__(self, policy_network, optimizer, env, gamma, max_steps, num_episodes, replay_buffer_size):
        self.policy_network = policy_network
        self.optimizer = optimizer
        self.env = env
        self.gamma = gamma
        self.max_steps = max_steps
        self.num_episodes = num_episodes

        self.replay_buffer = deque(maxlen=replay_buffer_size)

        self.episode_rewards_list = []
        self.episode_numbers_list = []
        self.steps_till_stop = []
        self.true_asset_price = []
        self.bid_prices = []
        self.ask_prices = []

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
    def loss_function(self, revenue, expenses, inventory, current_price, cash, only_profit=False):
        min_cash = 100
        max_inventory = 100

        # Compute profit and inventory value
        profit = revenue - expenses

        # Penalize low cash and high inventory
        cash_penalty = max(0, min_cash - cash)
        inventory_penalty = max(0, (inventory - max_inventory) * current_price)

        # Encourage profit
        profit_reward = -profit

        if only_profit:
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

            # Print episode information
            total_reward = sum(episode_rewards)

            print(f"Episode {episode + 1}, Total Reward: {total_reward}, Steps: {step}")

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
hidden_size = 20
num_episodes = 400
max_steps = 500
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

training = Training(policy_network, optimizer, env, gamma, max_steps, num_episodes, replay_buffer_size)
training.train()

env.close()

export = Export(training.episode_numbers_list, training.episode_rewards_list, training.steps_till_stop, training.true_asset_price, training.bid_prices, training.ask_prices)
export.plot_training()