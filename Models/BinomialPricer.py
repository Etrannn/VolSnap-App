import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import comb

class BinomialPricer:
    """
    A Vectorized Binomial Option Pricing Model for European and American options.
    
    Parameters:
    - S: Initial stock price
    - K: Option strike price
    - T: Time to maturity (in years)
    - r: Risk-free interest rate (annualized, in decimal)
    - sigma: Volatility of the underlying stock (annualized, in decimal)
    - steps: Number of time steps in the binomial tree
    - option_type: 'call' or 'put'
    - american: True for American options, False for European options

    Output:
    Option price for either call or put European or American
    """
    def __init__(self, S, K, T, r, sigma, steps=100, option_type='call', american="European"):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.steps = steps
        self.option_type = option_type
        self.american = american

    def price(self):
        dt = self.T / self.steps

        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u

        # Risk-neutral probability
        q = (np.exp(self.r * dt) - d) / (u - d)

        # Stock prices at maturity (final step)
        stock_prices = self.S * d ** np.arange(self.steps, -1, -1) * u ** np.arange(0, self.steps + 1)

        # Option values at maturity
        if self.option_type == 'call':
            option_values = np.maximum(0, stock_prices - self.K)
        elif self.option_type == 'put':
            option_values = np.maximum(0, self.K - stock_prices)
        else:
            raise ValueError("Invalid option type. Please specify 'call' or 'put'.")

        # Backward induction (vectorized)
        for i in range(self.steps - 1, -1, -1):
            option_values = np.exp(-self.r * dt) * (q * option_values[:-1] + (1 - q) * option_values[1:])
            if self.american == "American":
                stock_prices = stock_prices[:-1] * u  # Adjust stock prices for earlier time step
                if self.option_type == 'call':
                    option_values = np.maximum(option_values, stock_prices - self.K)
                elif self.option_type == 'put':
                    option_values = np.maximum(option_values, self.K - stock_prices)

        return option_values[0]
    
    

    def calculate_greeks(self):
        """
        Computes the Greeks for the Binomial model using finite differences.
        """
        eps = 1e-4
        base_price = BinomialPricer(self.S,self.K,self.T,self.r,self.sigma,self.steps,self.option_type,self.american).price()
        
        # Delta: ∂V/∂S
        price_up = BinomialPricer(self.S+eps,self.K,self.T,self.r,self.sigma,self.steps,self.option_type,self.american).price()
        price_down =BinomialPricer(self.S-eps,self.K,self.T,self.r,self.sigma,self.steps,self.option_type,self.american).price()
        delta = (price_up - price_down) / (2 * eps)

        # Gamma: ∂²V/∂S²
        gamma = (price_up - 2 * base_price + price_down) / (eps ** 2)

        # Vega: ∂V/∂σ
        price_vega = BinomialPricer(self.S,self.K,self.T,self.r,self.sigma+eps,self.steps,self.option_type,self.american).price()
        vega = (price_vega - base_price) / eps

        # Rho: ∂V/∂r (one-sided difference for better accuracy)
        price_rho = BinomialPricer(self.S,self.K,self.T,self.r+eps,self.sigma,self.steps,self.option_type,self.american).price()
        rho = (price_rho - base_price) / eps

        # Theta: ∂V/∂T (backward difference for better accuracy)
        price_theta = BinomialPricer(self.S,self.K,self.T-eps,self.r,self.sigma,self.steps,self.option_type,self.american).price()
        theta = (price_theta - base_price) / (-eps)
        
        data = {
            "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
            "Value": [delta, gamma, vega, theta, rho],
        }
        df = pd.DataFrame(data).set_index("Greek")
        return df

    def plot_stock_price_tree(self):
        """
        Visualizes the binomial tree of stock prices with lines connecting the nodes.
        """
        steps = 15
        dt = self.T / steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u

        # Build a tree (2D list) where tree[i][j] is the stock price at step i, node j.
        tree = [[0] * (i + 1) for i in range(steps + 1)]
        tree[0][0] = self.S
        for i in range(1, steps + 1):
            tree[i][0] = tree[i - 1][0] * u
            for j in range(1, i + 1):
                tree[i][j] = tree[i - 1][j - 1] * d

        plt.figure(figsize=(10, 6))
        # Plot the nodes and connect them
        for i in range(steps + 1):
            for j in range(i + 1):
                x = i  # x-coordinate corresponds to the time step
                y = tree[i][j]
                plt.scatter(x, y, color='blue')
                plt.text(x, y, f'{y:.2f}', fontsize=8, ha='center', va='bottom')

                # Draw lines from this node to the next step's nodes
                if i < steps:
                    # The two possible child nodes in the next step
                    child_up = tree[i + 1][j]      # upward move (multiplying by u)
                    child_down = tree[i + 1][j + 1]  # downward move (multiplying by d)
                    plt.plot([x, x + 1], [y, child_up], color='black', lw=0.8)
                    plt.plot([x, x + 1], [y, child_down], color='black', lw=0.8)

        plt.xlabel('Step')
        plt.ylabel('Stock Price')
        plt.title('Binomial Stock Price Tree')
        plt.grid(True)
        plt.show()

    def plot_option_price_tree(self):
        """
        Visualizes the binomial tree of option prices with lines connecting the nodes.
        """
        steps = 15
        dt = self.T / steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        q = (np.exp(self.r * dt) - d) / (u - d)

        # Initialize stock prices at maturity
        stock_prices = self.S * d ** np.arange(steps, -1, -1) * u ** np.arange(0, steps + 1)
        if self.option_type == 'call':
            option_values = np.maximum(0, stock_prices - self.K)
        else:
            option_values = np.maximum(0, self.K - stock_prices)

        # Build a tree for option values
        tree = [None] * (steps + 1)
        tree[steps] = list(option_values)

        for i in range(steps - 1, -1, -1):
            option_values = np.exp(-self.r * dt) * (q * option_values[:-1] + (1 - q) * option_values[1:])
            if self.american:
                # Recalculate stock prices for early exercise check.
                stock_prices = stock_prices[:-1] * u
                if self.option_type == 'call':
                    option_values = np.maximum(option_values, stock_prices - self.K)
                else:
                    option_values = np.maximum(option_values, self.K - stock_prices)
            tree[i] = list(option_values)

        plt.figure(figsize=(10, 6))
        # Plot the nodes and connect them
        for i in range(steps + 1):
            for j in range(i + 1):
                x = i
                y = tree[i][j]
                plt.scatter(x, y, color='red')
                plt.text(x, y, f'{y:.2f}', fontsize=8, ha='center', va='bottom')
                if i < steps:
                    # Connect to the two child nodes
                    child_up = tree[i + 1][j]
                    child_down = tree[i + 1][j + 1]
                    plt.plot([x, x + 1], [y, child_up], color='black', lw=0.8)
                    plt.plot([x, x + 1], [y, child_down], color='black', lw=0.8)
        plt.xlabel('Step')
        plt.ylabel('Option Price')
        plt.title('Binomial Option Price Tree with Connecting Lines')
        plt.grid(True)
        plt.show()

    def plot_volatility_impact(self):
        """
        Shows how volatility impacts the option price by plotting the option price as a function of volatility.
        """
        sigmas = np.linspace(0.1, 0.6, 10)
        prices = [BinomialPricer(self.S, self.K, self.T, self.r, self.sigma, self.steps, self.option_type, self.american).price() for self.sigma in sigmas]

        plt.figure(figsize=(8, 5))
        plt.plot(sigmas, prices, marker='o', color='green')
        plt.xlabel('Volatility')
        plt.ylabel('Option Price')
        plt.title('Impact of Volatility on Option Price')
        plt.grid(True)
        plt.show()

    def plot_terminal_dist(self):
        """
        Plots the probability distribution of terminal stock prices.
        """
        steps = 30
        dt = self.T / steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        q = (np.exp(self.r * dt) - d) / (u - d)

        # Compute probabilities for each terminal node
        probabilities = [comb(steps, i) * (q ** i) * ((1 - q) ** (steps - i)) for i in range(steps + 1)]
        stock_prices = self.S * d ** np.arange(steps, -1, -1) * u ** np.arange(0, steps + 1)

        plt.figure(figsize=(8, 5))
        sns.barplot(x=np.round(stock_prices, 2), y=probabilities, color='blue')
        plt.xlabel('Stock Price at Maturity')
        plt.ylabel('Probability')
        plt.title('Terminal Stock Price Distribution')
        plt.xticks(rotation=45)
        plt.show()

if __name__ == "__main__":
        
    # Example usage
    S = 100       # Current stock price
    K = 110       # Strike price
    T = 1         # Time to maturity (1 year)
    r = 0.08      # Risk-free interest rate (5%)
    sigma = 0.2   # Volatility (20%)
    steps = 1000   # Number of steps in the binomial tree

    # Price a European call option
    european_call = BinomialPricer(S, K, T, r, sigma, steps, option_type='call', american="European")
    print(f"European call option price: {european_call.price():.2f}")

    # Price an American put option
    american_call = BinomialPricer(S, K, T, r, sigma, steps, option_type='call', american="American")
    print(f"American put option price: {american_call.price():.2f}")

    # american_call.plot_stock_price_tree()
    # american_call.plot_option_price_tree()
    # american_call.plot_volatility_impact()
    # american_call.plot_terminal_dist()
    # print(european_call.calculate_greeks())
    

  
