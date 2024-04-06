import numpy as np

class MarketMakingEnvironment:
    def init(self, initial_inventory=100, spread=0.1):
        self.initial_inventory = initial_inventory
        self.inventory = initial_inventory
        self.spread = spread
        self.price = None
        self.observation_space = 2  # Dimensionality of the state space
        self.action_space = 1  # Dimensionality of the action space
        self.S0 = 100  #Initial price
        self.mu = 0.05  #Drift (expected return)
        self.sigma = 0.2 # Volatility
        self.dt = 1 / 252  #Time step (assuming daily updates for one year)


    
    def brownian_motion(S0, mu, sigma, dt):
        drift = mu * dt
        diffusion = sigma * np.sqrt(dt) * np.random.normal(0, 1)
        return S0 * np.exp(drift + diffusion)
    
    def reset(self):
        self.inventory = self.initial_inventory
        self.price = np.random.uniform(100, 200)
        return self.get_state()

    def step(self, action):
        # Apply the action (e.g., adjust bid/ask prices)
        bid_price = self.price - self.spread / 2
        ask_price = self.price + self.spread / 2

        # Execute trades and update inventory
        if action == 1:  # Buy
            self.inventory += 1
        elif action == -1:  # Sell
            self.inventory -= 1

        # Update price (e.g., random walk)
        self.price = self.brownian_motion(self.S0, self.mu, self.sigma , self.dt) # np.random.normal(0, 0.5)

        # Calculate reward (e.g., spread revenue)
        reward = bid_price - ask_price

        # Check if the episode is done (e.g., inventory depleted)
        done = self.inventory <= 0

        return self.get_state(), reward, done, {}

    def get_state(self):
        return np.array([self.inventory, self.price])