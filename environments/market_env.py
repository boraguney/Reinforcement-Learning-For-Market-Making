from matplotlib.pylab import f
import numpy as np
import torch

import gym
from gym import spaces
import ipdb

MAX_PRICE = 1000
MAX_INVENTORY = 1000
MAX_PROFIT = 100000
MAX_CASH = 1000000

class MarketEnv(gym.Env):
    # market is an instance of MarketSimulation
    # history is a dictionary containing: historical bid and ask, historical profits, previous buyer and seller
    def __init__(self, market, history, inventory, cash):
        self.total_profit = 0
        self.market = market

        self.hist_bid = history['bid']
        self.hist_ask = history['ask']
        self.hist_profit = history['profit']
        self.hist_inventory = history['inventory']
        self.hist_cash = history['cash']

        self.inventory = torch.tensor(inventory, requires_grad=True, dtype=torch.float32)
        self.cash = torch.tensor(cash, requires_grad=True, dtype=torch.float32)

        self.init_cash = cash
        self.init_inventory = inventory

        obs_low = np.array([0.0, 0.0] +
                          [0.0] * len(self.hist_bid) +
                          [0.0] * len(self.hist_ask) +
                          [0.0] * len(self.hist_profit) +
                          [0.0] * len(self.hist_inventory) +
                          [0.0] * len(self.hist_cash) +
                          [0.0, 0.0], dtype=np.float32) # current inventory and cash
        obs_high = np.array([MAX_PRICE, MAX_PRICE] +
                          [MAX_PRICE] * len(self.hist_bid) +
                          [MAX_PRICE] * len(self.hist_ask) +
                          [MAX_PROFIT] * len(self.hist_profit) +
                          [MAX_INVENTORY] * len(self.hist_inventory) +
                          [MAX_CASH] * len(self.hist_cash) +
                          [MAX_INVENTORY, MAX_CASH], dtype=np.float32)

        act_low = np.array([0.0, 0.0], dtype=np.float32)
        act_high = np.array([MAX_PRICE, MAX_PRICE], dtype=np.float32)

        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.state = None

    def compute_reward(self):
        net_profit = self.cash - 1000
        inventory_value = self.inventory * self.market.current_price

        total_reward = net_profit - inventory_value
        
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
        # Terminate if cash is zero or inventory is negative
        done = self.cash <= 0 or self.inventory < 0

        return self.state, reward, done, False, {}

    def reset(self, **kwargs):
        # Reset market state
        self.market.reset()

        self.inventory = torch.tensor(self.init_inventory, requires_grad=True, dtype=torch.float32)
        self.cash = torch.tensor(self.init_cash, requires_grad=True, dtype=torch.float32)

        # Initialize state with random bid and ask prices
        bid_price = np.random.uniform(0, MAX_PRICE)
        ask_price = np.random.uniform(0, MAX_PRICE)

        self.hist_bid = []
        self.hist_ask = []
        self.hist_profit = []
        self.hist_inventory = []
        self.hist_cash = []

        self.state = [bid_price, ask_price] + \
                        self.hist_bid + \
                        self.hist_ask + \
                        self.hist_profit + \
                        self.hist_inventory + \
                        self.hist_cash + \
                        [self.inventory, self.cash]
        self.state = torch.tensor(self.state, dtype=torch.float32)

        return self.state, {}

gym.register(id='MarketEnv', entry_point='environments.market_env:MarketEnv')