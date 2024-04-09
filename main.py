from collections import deque
import random

import gym

from matplotlib.style import available
from sympy import plot
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ipdb

from simulate import MarketSimulation

from environments.market_env import MarketEnv

# Define the neural network model
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, hidden_size // 8)
        self.fc5 = nn.Linear(hidden_size // 8, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class Plotting:
    def __init__(self, episode_numbers, episode_rewards):
        self.episode_numbers = episode_numbers
        self.episode_rewards = episode_rewards

    def plot_training(self):
        reg = np.polyfit(self.episode_numbers, self.episode_rewards, 1)
        
        plt.plot(self.episode_numbers, self.episode_rewards, label='Total Reward', color='blue')
        plt.plot(self.episode_numbers, np.polyval(reg, self.episode_numbers), label='Regression Line', color='red')
        plt.xlabel('Episode Number')
        plt.ylabel('Total Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
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

    # Function to select an action based on the policy network output
    def select_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        next_state = self.policy_network(state_tensor)
        next_state = torch.clamp(next_state, min=0, max=200)
        
        bid_price = next_state[0][0].item()
        ask_price = next_state[0][1].item()

        action = [bid_price, ask_price]

        return action

    # Function to compute the discounted rewards
    def compute_discounted_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards
    
    # Main loop
    def train(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            state = np.array(state[0])
            episode_rewards = []
            episode_states = []
            episode_actions = []
            
            for step in range(self.max_steps):
                action = self.select_action(state)
                next_state, reward, done, _, _= self.env.step(action)

                # Store experience in replay buffer
                self.replay_buffer.append((state, action, reward, next_state, done))

                episode_rewards.append(reward)
                episode_states.append(state)
                episode_actions.append(action)

                state = next_state
                
                if done:
                    break
            
            batch_size = min(len(self.replay_buffer), 32)  # Adjust batch size as needed
            batch = random.sample(self.replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.tensor(np.array(states))
            actions = torch.tensor(actions, dtype=torch.int64)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)
            
            discounted_rewards = self.compute_discounted_rewards(episode_rewards)
            discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-9)
            
            # Update policy network
            self.optimizer.zero_grad()
            for i in range(len(episode_rewards)):
                state = episode_states[i]
                action = episode_actions[i]
                reward = discounted_rewards[i]
                
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                output_tensor = self.policy_network(state_tensor)

                bid_price = output_tensor[0][0]
                ask_price = output_tensor[0][1]
                pred_tensor = torch.tensor([bid_price, ask_price, reward], dtype=torch.float32)

                max_revenue = self.env.market.get_max_revenue()
                min_expense = self.env.market.get_min_expenses()
                optimal_reward = max_revenue - min_expense
                optimal_bid_price = self.env.market.get_optimal_bid_price()
                optimal_ask_price = self.env.market.get_optimal_ask_price()

                target_tensor = torch.tensor([optimal_bid_price, optimal_ask_price, optimal_reward], dtype=torch.float32, requires_grad=True)

                loss_function = nn.MSELoss()
                loss = loss_function(pred_tensor, target_tensor)
                loss.backward()

            self.optimizer.step()

            # store episode rewards and episode number
            self.episode_rewards_list.append(sum(episode_rewards))
            self.episode_numbers_list.append(episode + 1)
            
            # Print episode information
            total_reward = sum(episode_rewards)
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
hidden_size = 10
num_episodes = 200
max_steps = 200
replay_buffer_size = 1000

market = MarketSimulation()

# Initialize environment and policy network
env = gym.make('MarketEnv', market=market)
input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]
policy_network = PolicyNetwork(input_size, hidden_size, output_size)
optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)

training = Training(policy_network, optimizer, env, gamma, max_steps, num_episodes, replay_buffer_size)
training.train()

env.close()

plotting = Plotting(training.episode_numbers_list, training.episode_rewards_list)
plotting.plot_training()