import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# Black-Scholes price formula
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Function to calculate implied volatility
def implied_volatility(S, K, T, r, market_price, option_type="call"):
    func = lambda sigma: black_scholes(S, K, T, r, sigma, option_type) - market_price
    return brentq(func, 1e-5, 5)

# Example data
S = 100  # Spot price
r = 0.05  # Risk-free rate
market_prices = np.array([
    [10, 15, 20, 18],  # Prices for 30 days to expiration
    [11, 16, 21, 19],  # Prices for 60 days to expiration
    [12, 17, 22, 20]   # Prices for 90 days to expiration
])
strikes = np.array([90, 100, 110, 120])  # Strike prices
maturities = np.array([30, 60, 90])  # Maturities in days

# Calculate implied volatilities
implied_vols = np.zeros_like(market_prices)

for i in range(len(maturities)):
    for j in range(len(strikes)):
        implied_vols[i, j] = implied_volatility(S, strikes[j], maturities[i] / 365, r, market_prices[i, j])

# Create a 3D plot
X, Y = np.meshgrid(strikes, maturities)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, implied_vols, cmap="viridis")

# Labels
ax.set_xlabel('Strike Price')
ax.set_ylabel('Time to Maturity (days)')
ax.set_zlabel('Implied Volatility')
ax.set_title('Implied Volatility Surface')

plt.show()
