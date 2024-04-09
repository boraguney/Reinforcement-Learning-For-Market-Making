import numpy as np
import matplotlib.pyplot as plt
from pyparsing import line

class MarketSimulation:
    def __init__(self,
        initial_price=50, volatility=5, time_step=1,
        buyer_count=5, buyer_scale=0.02,
        seller_count=5,seller_scale=0.02):
        self.initial_price = initial_price

        self.volatility = volatility
        self.time_step = time_step

        self.current_price = initial_price

        self.buyer_count = buyer_count
        self.buyer_scale = buyer_scale
        self.current_buyer_maximums = self.buyer_maximum_prices()

        self.seller_count = seller_count
        self.seller_scale = seller_scale
        self.current_seller_minimums = self.seller_minimum_prices()

    def next_price(self):
        increment = np.random.normal(0, self.volatility * np.sqrt(self.time_step))
        self.current_price += increment
        self.current_buyer_maximums = self.buyer_maximum_prices()
        self.current_seller_minimums = self.seller_minimum_prices()
        return self.current_price
    
    # Calculates the maximum revenue obtainable given the current buyer/seller prices
    def get_max_revenue(self):
        buyer_max = np.max(self.current_buyer_maximums)
        return self.get_revenue(buyer_max)
    
    def get_min_expenses(self):
        seller_min = np.min(self.current_seller_minimums)
        return self.get_expenses(seller_min)

    def get_revenue(self, bid_price):
        # number of units sold * price
        return bid_price * self.get_agent_purchases(bid_price)

    def get_expenses(self, ask_price):
        # number of units purchased * price
        return ask_price * self.get_agent_sales(ask_price)

    def get_agent_purchases(self, bid_price):
        # agent declares they will buy at bid price
        # sellers will sell if it's above their minimum
        return np.sum(bid_price >= self.current_seller_minimums)

    def get_agent_sales(self, ask_price):
        # agent declares they will sell at ask price
        # buyers will purchase if it's below their maximum
        return np.sum(ask_price <= self.current_buyer_maximums)

    def buyer_maximum_prices(self):
        return self.current_price * np.random.normal(1, self.buyer_scale, size=self.buyer_count)
    
    def get_optimal_bid_price(self):
        return np.max(self.current_buyer_maximums)
    
    def get_optimal_ask_price(self):
        return np.min(self.current_seller_minimums)

    def seller_minimum_prices(self):
        return self.current_price * np.random.normal(1, self.seller_scale, size=self.seller_count)
    
    def reset(self):
        self.current_price = self.initial_price