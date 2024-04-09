import numpy as np

import gym
from gym import spaces
import ipdb

class MarketEnv(gym.Env):
    def __init__(self, market):
        self.market = market
        
        low = np.array([0.0, 0.0], dtype=np.float32)
        high = np.array([200.0, 200.0], dtype=np.float32)
        
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.state = None

    def step(self, action):        
        bid_price, ask_price = action

        revenue = self.market.get_revenue(bid_price)
        expenses = self.market.get_expenses(ask_price)
        reward = revenue - expenses

        self.market.current_price = self.market.next_price()
        self.market.current_buyer_maximums = self.market.buyer_maximum_prices()
        self.market.current_seller_minimums = self.market.seller_minimum_prices()

        # Update state
        self.state = [bid_price, ask_price]

        # Termination condition
        done = False

        return np.array(self.state, dtype=np.float32), reward, done, False, {}

    def reset(self, **kwargs):
        # Reset market state
        self.market.reset()

        # Initialize state with random bid and ask prices
        bid_price = np.random.uniform(0, 200)
        ask_price = np.random.uniform(0, 200)
        self.state = [bid_price, ask_price]

        return np.array(self.state, dtype=np.float32), {}

gym.register(id='MarketEnv', entry_point='environments.market_env:MarketEnv')