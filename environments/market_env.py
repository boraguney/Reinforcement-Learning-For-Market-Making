from matplotlib.pylab import f
import numpy as np
import torch

import gym
from gym import spaces
import ipdb

class MarketEnv(gym.Env):
    # market is an instance of MarketSimulation
    # history is a dictionary containing: historical bid and ask, historical profits, previous buyer and seller
    def __init__(self, market, history, inventory, cash):
        self.total_profit = 0
        self.market = market

        self.hist_bid = history['bid']
        self.hist_ask = history['ask']
        self.hist_profit = history['profit']

        self.inventory = torch.tensor(inventory, requires_grad=True, dtype=torch.float32)
        self.cash = torch.tensor(cash, requires_grad=True, dtype=torch.float32)

        self.init_cash = cash
        self.init_inventory = inventory

        obs_low = np.array([0.0, 0.0] +
                          [0.0] * len(self.hist_bid) +
                          [0.0] * len(self.hist_ask) +
                          [0.0] * len(self.hist_profit) +
                          [0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1000.0, 1000.0] +
                          [1000.0] * len(self.hist_bid) +
                          [1000.0] * len(self.hist_ask) +
                          [100000.0] * len(self.hist_profit) +
                          [100000.0, 100000.0], dtype=np.float32)
        
        act_low = np.array([0.0, 0.0], dtype=np.float32)
        act_high = np.array([1000.0, 1000.0], dtype=np.float32)

        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.state = None

    def compute_reward(self):
        net_profit = self.cash - 1000
        inventory_reward = self.inventory * self.market.current_price

        total_reward = net_profit + inventory_reward
        
        return total_reward

    def step(self, action):
        bid_price, ask_price = action

        sales = self.market.get_sales(ask_price)
        purchases = self.market.get_purchases(bid_price)

        revenue = self.market.get_revenue(ask_price)
        expenses = self.market.get_expenses(bid_price)

        change_in_inventory = purchases - sales
        self.inventory = self.inventory + change_in_inventory
        self.cash = self.cash + revenue - expenses

        reward = self.compute_reward()

        self.market.current_price = self.market.next_price()

        # print(f"Bid price: {bid_price} \nAsk price: {ask_price}")
        # print(f"Price: {self.market.current_price} \nInventory: {self.inventory} \nCash: {self.cash} \nReward: {reward}")

        # Update state
        self.hist_bid.append(bid_price)
        self.hist_ask.append(ask_price)
        self.hist_profit.append(reward)

        self.state = [bid_price, ask_price] + \
                        self.hist_bid + \
                        self.hist_ask + \
                        self.hist_profit + \
                        [self.inventory, self.cash]
        self.state = torch.tensor(self.state, dtype=torch.float32)

        # Termination condition
        done = False

        return self.state, reward, done, False, {}

    def reset(self, **kwargs):
        # Reset market state
        self.market.reset()

        # Initialize state with random bid and ask prices
        bid_price = np.random.uniform(0, 1000)
        ask_price = np.random.uniform(0, 1000)


        self.inventory = torch.tensor(self.init_inventory, requires_grad=True, dtype=torch.float32)
        self.cash = torch.tensor(self.init_cash, requires_grad=True, dtype=torch.float32)

        self.hist_bid = [bid_price]
        self.hist_ask = [ask_price]
        self.hist_profit = [0.0]

        self.state = [bid_price, ask_price] + \
                        self.hist_bid + \
                        self.hist_ask + \
                        self.hist_profit + \
                        [self.inventory, self.cash]
        self.state = torch.tensor(self.state, dtype=torch.float32)

        return self.state, {}

gym.register(id='MarketEnv', entry_point='environments.market_env:MarketEnv')