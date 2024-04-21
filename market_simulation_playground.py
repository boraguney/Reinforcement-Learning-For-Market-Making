from simulate import MarketSimulation
import matplotlib.pyplot as plt

m = MarketSimulation()

asset_prices = []
all_bid_prices = []
all_ask_prices = []

for i in range(100):
    asset_prices.append(m.next_price())
    current_bid_prices = m.current_buyer_maximums
    current_ask_prices = m.current_seller_minimums
    all_bid_prices.append(current_bid_prices)
    all_ask_prices.append(current_ask_prices)

plt.plot(asset_prices, linewidth=4, color='black')

for i, (bid_prices, ask_prices) in enumerate(zip(all_bid_prices, all_ask_prices)):
    x_values = [i] * len(bid_prices)
    plt.plot(x_values, bid_prices, 'ro', alpha=0.5, markersize=2.5)
    plt.plot(x_values, ask_prices, 'bo', alpha=0.5, markersize=2.5)

plt.legend(['Asset Price', 'Bid Prices', 'Ask Prices'])
plt.show()