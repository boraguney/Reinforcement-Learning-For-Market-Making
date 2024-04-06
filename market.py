import numpy as np

class Market:
    def init(self,
        initial_price=100, expected_walk=0, scale_walk=0.1,
        buyer_count=10, buyer_scale=0.1,
        seller_count=10,seller_scale=0.1):
        self.initial_price = initial_price
        self.change_expectation = expected_walk
        self.change_scale = scale_walk

        self.current_price = initial_price

        self.buyer_count = buyer_count
        self.buyer_scale = buyer_scale
        self.current_buyer_maximums = np

        self.seller_count = self.seller_minimum_prices()
        self.seller_scale = seller_scale
        self.current_seller_maximums = np

    def next_price(self):
        walk = np.random.normal(self.change_expectation, self.change_scale)
        self.current_price = self.current_price * walk
        self.current_buyer_maximums = self.buyer_maximum_prices()
        self.current_seller_minimums = self.seller_minimum_prices()
        return self.current_price

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

    def seller_minimum_prices(self):
        return self.current_price * np.random.normal(1, self.seller_scale, size=self.seller_count)
    
    def reset(self):
        self.current_price = self.initial_price

