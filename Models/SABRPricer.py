import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt


class SABRPricer:
    """
    Closed Solution
    MonteCarlo
    """
    def __init__(self, Asset_info, Process_info):
        self.nu = Process_info.nu 
        self.alpha = Process_info.alpha  
        self.beta = Process_info.beta
        self.rho = Process_info.rho  

        
        self.S0 = Asset_info.S0
        self.K = Asset_info.K
        self.r = Process_info.r 
        self.T = Asset_info.T
        self.notional = Asset_info.notional 
        self.asset_type = Asset_info.asset_type  

        self.F = self.S0 * np.exp(self.r * self.T) 
        
             
        

        self.price = 0
        self.S_vec = None
        self.price_vec = None
        self.mesh = None
        self.payoff = Asset_info.payoff
    
    def payoff_f(self, S):
        if self.payoff == "call":
            Payoff = np.maximum(S - self.K, 0)
        elif self.payoff == "put":
            Payoff = np.maximum(self.K - S, 0)
        return Payoff
    

    def implied_vol(self):
        """
        Computes the implied volatility using the SABR model approximation.
        Equation from Hagan et. al. paper
        
        Parameters:
        F : float   -> Forward price
        K : float   -> Strike price
        T : float   -> Time to maturity
        alpha : float -> SABR parameter (volatility of volatility)
        beta : float  -> SABR parameter (elasticity)
        rho : float   -> Correlation between asset and volatility
        vol_of_vol : float -> Volatility of volatility
        
        Returns:
        float -> Implied volatility
        """
        if self.T * self.nu**2 > 1:
            print("Warning: High value of time vol^2 condition may result in inaccurate implied volatility")


        if self.F == self.K:
            FK = self.F
        else:
            FK = np.sqrt(self.F * self.K)
        
        z = self.nu / self.alpha * (self.F * self.K)**((1 - self.beta) / 2)  * np.log(self.F / self.K)
        x_z = np.log((np.sqrt(1 - 2 * self.rho * z + z**2) + z - self.rho) / (1 - self.rho))
        
        A = self.alpha / (FK ** (1 - self.beta))
        
        B = 1 + ((1 - self.beta)**2 / 24 * (np.log(self.F / self.K))**2 + 
                 (1 - self.beta)**4 / 1920 * (np.log(self.F / self.K))**4)
        
        C = 1 + self.T * ((1 - self.beta)**2 / 24 * (self.alpha**2 / (FK**(2 * (1 - self.beta)))) + 
                          self.rho * self.beta * self.nu / (4 * FK**(1 - self.beta)) + 
                          (2 - 3 * self.rho**2) / 24 * self.nu**2)
        
        if self.F == self.K:
            implied_vol = A * C
        else:
            implied_vol = A * (z / x_z) * B * C
        
        return implied_vol
    
    def black_formula(self):
        """
        Used to price future contracts, bond options, interest rate cap and floors, and swaptions.
        """
        sigma = self.implied_vol()
        if self.asset_type == "interest":
            d1 = (np.log(self.F / self.K) + 0.5 * sigma**2 * self.T) / (sigma * np.sqrt(self.T))
            d2 = d1 - sigma * np.sqrt(self.T)
            discount_factor = np.exp(-self.r * self.T)

            if self.payoff == "call":
                return discount_factor * self.notional * (self.F * ss.norm.cdf(d1) - self.K * ss.norm.cdf(d2))
            elif self.payoff == "put":
                return self.K * ss.norm.cdf(-d2) - discount_factor * self.notional * (self.F * ss.norm.cdf(-d1))
            else:
                raise ValueError("invalid type. Set 'call' or 'put'")
                    
        elif self.asset_type == "equity":
            d1 = (np.log(self.S0 / self.K) + (self.r + sigma**2 / 2) * self.T) / (sigma * np.sqrt(self.T))
            d2 = (np.log(self.S0 / self.K) + (self.r - sigma**2 / 2) * self.T) / (sigma * np.sqrt(self.T))

            if self.payoff == "call":
                return self.S0 * ss.norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * ss.norm.cdf(d2)
            elif self.payoff == "put":
                return self.K * np.exp(-self.r * self.T) * ss.norm.cdf(-d2) - self.S0 * ss.norm.cdf(-d1)
            else:
                raise ValueError("invalid type. Set 'call' or 'put'")
        
    def black_formula_vector(self, K):
        """
        Used to price future contracts, bond options, interest rate caps and floors, and swaptions.
        
        Parameters:
            K (array-like): Array of strike prices.
        
        Returns:
            np.ndarray: Array of option prices for each strike price in K.
        """
        sigma = self.implied_vol()
        K = np.array(K)  # Ensure K is a NumPy array for vectorization
        
        if self.asset_type == "interest":
            d1 = (np.log(self.F / K) + 0.5 * sigma**2 * self.T) / (sigma * np.sqrt(self.T))
            d2 = d1 - sigma * np.sqrt(self.T)
            discount_factor = np.exp(-self.r * self.T)

            if self.payoff == "call":
                prices = discount_factor * self.notional * (self.F * ss.norm.cdf(d1) - K * ss.norm.cdf(d2))
            elif self.payoff == "put":
                prices = discount_factor * self.notional * (K * ss.norm.cdf(-d2) - self.F * ss.norm.cdf(-d1))
            else:
                raise ValueError("Invalid type. Set 'call' or 'put'")

        elif self.asset_type == "equity":
            d1 = (np.log(self.S0 / K) + (self.r + sigma**2 / 2) * self.T) / (sigma * np.sqrt(self.T))
            d2 = d1 - sigma * np.sqrt(self.T)

            if self.payoff == "call":
                prices = self.S0 * ss.norm.cdf(d1) - K * np.exp(-self.r * self.T) * ss.norm.cdf(d2)
            elif self.payoff == "put":
                prices = K * np.exp(-self.r * self.T) * ss.norm.cdf(-d2) - self.S0 * ss.norm.cdf(-d1)
            else:
                raise ValueError("Invalid type. Set 'call' or 'put'")

        else:
            raise ValueError("Invalid asset type. Set 'interest' or 'equity'")

        return prices


    def priceSurface(self, K_range, T_range, num_K=10, num_T=10):
            """
            Generate a 3D surface plot for option prices under the Merton Jump Diffusion model.
            
            Parameters:
            - num_K: Number of strike price points.
            - num_T: Number of maturity points.        
            - K_range: Tuple (K_min, K_max) for strike prices.
            - T_range: Tuple (T_min, T_max) for time to maturity.

            """
            K_values = np.linspace(K_range[0], K_range[1], num_K)
            T_values = np.linspace(T_range[0], T_range[1], num_T)
            K_grid, T_grid = np.meshgrid(K_values, T_values)
            
            price_surface = np.zeros_like(K_grid)
            
            for i in range(num_K):
                for j in range(num_T):
                    self.K = K_values[i]
                    self.T = T_values[j]
                    price_surface[j, i] = self.black_formula_vector(K)
            
            # Plotting the surface
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(K_grid, T_grid, price_surface, cmap='viridis')
            
            ax.set_xlabel('Strike Price (K)')
            ax.set_ylabel('Time to Maturity (T)')
            ax.set_zlabel('Option Price')
            ax.set_title(f'SABR Model {self.payoff.capitalize()} Option Price Surface')
            plt.show()

    def plot_IV_surface(self, K_range, T_range,num_K=10,num_T=10):
        """
        Plot IV surface for different strikes and maturities.
        
        - K_range: Range of strike prices.
        - T_range: Range of maturities (in years).
        - num_K: Number of strike price points.
        - num_T: Number of maturity points.
        
        """
        K_values = np.linspace(K_range[0], K_range[1], num_K)
        T_values = np.linspace(T_range[0], T_range[1], num_T)
        K_grid, T_grid = np.meshgrid(K_values, T_values)
        
        price_surface = np.zeros_like(K_grid)
        
        for i in range(num_K):
            for j in range(num_T):
                self.K = K_values[i]
                self.T = T_values[j]
                price_surface[j, i] = self.implied_vol()
        
        # Plotting the surface
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(K_grid, T_grid, price_surface, cmap='viridis')
        
        ax.set_xlabel('Strike Price (K)')
        ax.set_ylabel('Time to Maturity (T)')
        ax.set_zlabel('Implied Volatility')
        ax.set_title(f'SABR Implied Volatility Surface')
        plt.show()



    def visualisePaths(self,N=1000, num_simulations = 10000):
        """
        N: Number of time steps
        """
        # Parameters
        S0 = self.S0        # Initial asset price
        alpha0 = self.alpha    # Initial volatility
        beta = self.beta      # Beta parameter in SABR model
        rho = self.rho      # Correlation between asset and volatility
        nu = self.nu        # Volatility of volatility
        T = self.T         # Time to maturity (1 year)
        dt = T / N      # Time step size
        


        # Initialize arrays to store results
        S_paths = np.zeros((num_simulations, N + 1))
        alpha_paths = np.zeros((num_simulations, N + 1))

        # Set initial values
        S_paths[:, 0] = S0
        alpha_paths[:, 0] = alpha0

        # Generate random normal variables for Z1, Z2, and Z_tilde
        Z1 = np.random.normal(0, 1, (num_simulations, N))
        Z2 = np.random.normal(0, 1, (num_simulations, N))
        Z_tilde = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        # Vectorized Monte Carlo simulation
        for n in range(N):
            # Update asset price paths
            S_paths[:, n + 1] = S_paths[:, n] + alpha_paths[:, n] * S_paths[:, n]**beta * np.sqrt(dt) * Z1[:, n]
            
            # Update volatility paths
            alpha_paths[:, n + 1] = alpha_paths[:, n] + nu * alpha_paths[:, n] * np.sqrt(dt) * Z_tilde[:, n]

        # Compute averages across all simulations
        S_mean = np.mean(S_paths, axis=0)
        alpha_mean = np.mean(alpha_paths, axis=0)


        # Plot the results
        plt.figure(figsize=(12, 6)) 
        plt.subplot(1, 2, 1)

        plt.plot(S_paths[:50, :].T)  # Show first 10 paths
        plt.title('SABR Model Forward Price Paths')
        plt.xlabel('Time Steps')
        plt.ylabel('Forward Price')

        plt.subplot(1, 2, 2)
        plt.plot(alpha_paths[:50, :].T)  # Show first 10 paths
        plt.title('SABR Model Variance Process')
        plt.xlabel('Time Steps')
        plt.ylabel('Variance')
        plt.tight_layout()
        plt.show()

class asset_param:
        def __init__(self, S0, K, T,notional = 0, asset_type="interest", payoff="call"):
            self.S0 = S0  # Initial stock price
            self.K = K  # Strike price
            self.T = T  # Time to maturity
            self.notional = notional
                        
            if asset_type == "equity" or asset_type == "interest":
                self.asset_type = asset_type
            else:
                raise ValueError("invalid type. Set 'equity' or 'interest'")

            if payoff == "call" or payoff == "put":
                self.payoff = payoff
            else:
                raise ValueError("invalid type. Set 'call' or 'put'")
           

class process_info:
    def __init__(self, r, nu, alpha, beta, rho):
        self.r = r  
        self.nu = nu  
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        




if __name__ == "__main__":
    S0 = 100
    K = 110
    T = 1.0
    alpha = 0.8
    beta = 0.4
    rho = -0.7
    nu = 0.25
    r = 0.05
    
    assetParam = asset_param(S0,K,T,notional=1_500_000,asset_type='equity',payoff='call')
    processInfo = process_info(r,nu,alpha,beta,rho)

    pricer = SABRPricer(assetParam,processInfo)
    print(pricer.black_formula())
    
    strikes = [60,180]
    maturities = [0.01,3]
    pricer.priceSurface(strikes,maturities,25,25)
    pricer.plot_IV_surface(strikes,maturities,25,25)

    pricer.visualisePaths()



