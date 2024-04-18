import numpy as np
import matplotlib.pyplot as plt
from pyparsing import line
import torch

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



    def get_revenue(self, ask_price):
        # number of units sold * price
        return ask_price * self.get_sales(ask_price)

    def get_expenses(self, bid_price):
        # number of units purchased * price
        return bid_price * self.get_purchases(bid_price)
    



    def get_sales(self, ask_price):
        # agent declares they will sell at ask price
        # buyers will purchase if it's below their maximum
        return torch.sum(ask_price <= torch.tensor(self.current_buyer_maximums, requires_grad=False))

    def get_purchases(self, bid_price):
        # number of units purchased
        # agent declares they will buy at bid price
        # sellers will sell if it's above their minimum
        return torch.sum(bid_price >= torch.tensor(self.current_seller_minimums, requires_grad=False))




    def buyer_maximum_prices(self):
        return self.current_price * np.random.normal(1, self.buyer_scale, size=self.buyer_count)

    def seller_minimum_prices(self):
        return self.current_price * np.random.normal(1, self.seller_scale, size=self.seller_count)
    
    def reset(self):
        self.current_price = self.initial_price