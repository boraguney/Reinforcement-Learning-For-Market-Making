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

class TrainingSimpleModel:
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
        self.episode_agent = []
        self.steps_till_stop = []

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
    def loss_function(self, profit):
        loss = -profit

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

            #discounted_rewards = self.compute_discounted_rewards(episode_rewards)
            discounted_rewards = episode_rewards
            #discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-9)

            # Update policy network
            self.optimizer.zero_grad()
            state = episode_states[-1]
            action = episode_actions[-1]
            reward = episode_rewards[-1]

            self.policy_network.adjust_input_size(state.size()[0])
            output_tensor = self.policy_network(state)

            inventory = state[-2]

            loss = self.loss_function(reward)
            loss.backward(retain_graph=True)
            self.optimizer.step()

            # store episode rewards and episode number
            self.episode_rewards_list.append(reward)
            self.episode_numbers_list.append(episode + 1)

            # Print episode information
            
            print(f"Episode {episode + 1}, Total Reward: {reward}")
