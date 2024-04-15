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
        self.hidden_size = hidden_size
        self.output_size = output_size

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

    def adjust_input_size(self, new_input_size):
        self.fc1 = nn.Linear(new_input_size, self.hidden_size)

class Plotting:
    def __init__(self, episode_numbers, episode_rewards):
        self.episode_numbers = episode_numbers
        self.episode_rewards = episode_rewards

    def plot_training(self):
        detached_rewards = [reward.detach().numpy() for reward in self.episode_rewards]
        numpy_rewards = np.array(detached_rewards)

        plt.plot(self.episode_numbers, numpy_rewards, label='Total Reward', color='blue')
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
        self.policy_network.adjust_input_size(state.size()[0])
        next_state = self.policy_network(state)
        next_state = torch.clamp(next_state, min=0, max=1000)

        return next_state

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
    def loss_function(self, bid, ask, profit):
        loss = -profit**2

        return loss

    # Main loop
    def train(self):
        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            episode_rewards = []
            episode_states = []
            episode_actions = []
            
            for step in range(self.max_steps):
                action = self.select_action(state)
                next_state, reward, done, _, _= self.env.step(action)

                episode_rewards.append(reward)
                episode_states.append(state)
                episode_actions.append(action)

                state = next_state

                if done:
                    break

            discounted_rewards = self.compute_discounted_rewards(episode_rewards)
            #discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-9)

            # Update policy network
            self.optimizer.zero_grad()
            for i in range(len(episode_rewards)):
                state = episode_states[i]
                action = episode_actions[i]
                reward = discounted_rewards[i]

                self.policy_network.adjust_input_size(state.size()[0])
                output_tensor = self.policy_network(state)

                bid = output_tensor[0]
                ask = output_tensor[1]

                loss = self.loss_function(bid, ask, reward)
                loss.backward(retain_graph=True)

            self.optimizer.step()

            # store episode rewards and episode number
            self.episode_rewards_list.append(sum(episode_rewards))
            self.episode_numbers_list.append(episode + 1)

            # Print episode information
            total_reward = sum(episode_rewards)
            
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Hyperparameters
learning_rate = 0.001
gamma = 0.8
hidden_size = 50
num_episodes = 500
max_steps = 20
replay_buffer_size = 1000

market = MarketSimulation()

# Initialize environment and policy network

history = {
    'bid': [],
    'ask': [],
    'profit': [],
    'buyer': [],
    'seller': []
}

env = gym.make('MarketEnv', market=market, history=history, inventory=0, cash=1000)
input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]
policy_network = PolicyNetwork(input_size, hidden_size, output_size)
optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)

training = Training(policy_network, optimizer, env, gamma, max_steps, num_episodes, replay_buffer_size)
training.train()

env.close()

plotting = Plotting(training.episode_numbers_list, training.episode_rewards_list)
plotting.plot_training()