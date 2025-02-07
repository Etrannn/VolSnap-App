import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from functools import partial
import time 
import pandas as pd
from prereq import fft_Lewis, IV_from_Lewis, Q1, Q2


def cf_Heston_good(u, t, v0, mu, kappa, theta, sigma, rho):
    """
    Heston characteristic function as proposed by Schoutens (2004)
    """
    xi = kappa - sigma * rho * u * 1j
    d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
    g1 = (xi + d) / (xi - d)
    g2 = 1 / g1
    cf = np.exp(
        1j * u * mu * t
        + (kappa * theta) / (sigma**2) * ((xi - d) * t - 2 * np.log((1 - g2 * np.exp(-d * t)) / (1 - g2)))
        + (v0 / sigma**2) * (xi - d) * (1 - np.exp(-d * t)) / (1 - g2 * np.exp(-d * t))
    )
    return cf

class Heston_pricer:
    """
    Class to price the options with the Heston model by:
    - Fourier-inversion.
    - Monte Carlo.
    """

    def __init__(self, Option_info, Process_info):
        """
        Process_info:  of type VG_process. It contains the interest rate r
        and the VG parameters (sigma, theta, kappa)

        Option_info:  of type Option_param. It contains (S0,K,T) i.e. current price,
        strike, maturity in years
        """
        self.r = Process_info.mu  # interest rate
        self.sigma = Process_info.sigma  # Heston parameter
        self.theta = Process_info.theta  # Heston parameter
        self.kappa = Process_info.kappa  # Heston parameter
        self.rho = Process_info.rho  # Heston parameter

        self.S0 = Option_info.S0  # current price
        self.v0 = Option_info.v0  # spot variance
        self.K = Option_info.K  # strike
        self.T = Option_info.T  # maturity in years

        self.exercise = Option_info.exercise
        self.payoff = Option_info.payoff
    
    
    def FI_for_Greeks(payoff, S0, K,T,v0,r,theta,sigma,kappa,rho):
        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(K / S0)  # log moneyness
        cf_H_b_good = partial(
            cf_Heston_good,
            t=T,
            v0=v0,
            mu=r,
            theta=theta,
            sigma=sigma,
            kappa=kappa,
            rho=rho,
        )

        limit_max = 2000  # right limit in the integration

        if payoff == "call":
            call = S0 * Q1(k, cf_H_b_good, limit_max) - K * np.exp(-r * T) * Q2(
                k, cf_H_b_good, limit_max
            )
            return call
        elif payoff == "put":
            put = K * np.exp(-r * T) * (1 - Q2(k, cf_H_b_good, limit_max)) - S0 * (
                1 - Q1(k, cf_H_b_good, limit_max)
            )
            return put
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")


    def payoff_f(self, S):
        if self.payoff == "call":
            Payoff = np.maximum(S - self.K, 0)
        elif self.payoff == "put":
            Payoff = np.maximum(self.K - S, 0)
        return Payoff
    
    def Fourier_inversion(self):
        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(self.K / self.S0)  # log moneyness
        cf_H_b_good = partial(
            cf_Heston_good,
            t=self.T,
            v0=self.v0,
            mu=self.r,
            theta=self.theta,
            sigma=self.sigma,
            kappa=self.kappa,
            rho=self.rho,
        )

        limit_max = 2000  # right limit in the integration

        if self.payoff == "call":
            call = self.S0 * Q1(k, cf_H_b_good, limit_max) - self.K * np.exp(-self.r * self.T) * Q2(
                k, cf_H_b_good, limit_max
            )
            return call
        elif self.payoff == "put":
            put = self.K * np.exp(-self.r * self.T) * (1 - Q2(k, cf_H_b_good, limit_max)) - self.S0 * (
                1 - Q1(k, cf_H_b_good, limit_max)
            )
            return put
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def FFT(self, K):
        """
        FFT method. It returns a vector of prices.
        K is an array of strikes
        """
        K = np.array(K)
        cf_H_b_good = partial(
            cf_Heston_good,
            t=self.T,
            v0=self.v0,
            mu=self.r,
            theta=self.theta,
            sigma=self.sigma,
            kappa=self.kappa,
            rho=self.rho,
        )

        if self.payoff == "call":
            return fft_Lewis(K, self.S0, self.r, self.T, cf_H_b_good, interp="cubic")
        elif self.payoff == "put":  # put-call parity
            return (
                fft_Lewis(K, self.S0, self.r, self.T, cf_H_b_good, interp="cubic")
                - self.S0
                + K * np.exp(-self.r * self.T)
            )
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")


    def MC(self, N, paths, Err=False, Time=False):
        """
        Heston Monte Carlo simulation.

        Parameters:
            N     : number of time steps per simulation.
            paths : number of simulated paths.
            Err   : if True, return standard error.
            Time  : if True, return execution time.
        """
        t_init = time.time()
        hestonprocess = Heston_process(mu=self.r, rho=self.rho, sigma=self.sigma, theta=self.theta, kappa=self.kappa)
        
        # Collect terminal asset prices from each simulation.
        terminal_prices = np.zeros(paths)
        for i in range(paths):
            S_path, _ = hestonprocess.path(S0=self.S0, v0=self.v0, N=N, T=self.T)
            terminal_prices[i] = S_path[-1]  # get the terminal price from each path

        # Reshape to a 2D array if needed by your payoff function.
        S_T = terminal_prices.reshape((paths, 1))
        
        # Compute the discounted payoff.
        DiscountedPayoff = np.exp(-self.r * self.T) * self.payoff_f(S_T)
        V = np.mean(DiscountedPayoff, axis=0)
        std_err = ss.sem(DiscountedPayoff)

        if Err:
            if Time:
                elapsed = time.time() - t_init
                return V, std_err, elapsed
            else:
                return V, std_err
        else:
            if Time:
                elapsed = time.time() - t_init
                return V, elapsed
            else:
                return V

    def IV_Lewis(self):
        """Implied Volatility from the Lewis formula"""

        cf_H_b_good = partial(
            cf_Heston_good,
            t=self.T,
            v0=self.v0,
            mu=self.r,
            theta=self.theta,
            sigma=self.sigma,
            kappa=self.kappa,
            rho=self.rho,
        )
        if self.payoff == "call":
            return IV_from_Lewis(self.K, self.S0, self.T, self.r, cf_H_b_good)
        elif self.payoff == "put":
            raise NotImplementedError
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")
        
    def priceSurface(self, K_range, T_range, num_K=10, num_T=10):
        """
        Generate a 3D surface plot for option prices under the Heston model.
        
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
                price_surface[j, i] = self.FFT(K)
        
        # Plotting the surface
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(K_grid, T_grid, price_surface, cmap='viridis')
        
        ax.set_xlabel('Strike Price (K)')
        ax.set_ylabel('Time to Maturity (T)')
        ax.set_zlabel('Option Price')
        ax.set_title(f'Heston Model {self.payoff.capitalize()} Option Price Surface')
        
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
                price_surface[j, i] = self.IV_Lewis()
        
        # Plotting the surface
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(K_grid, T_grid, price_surface, cmap='viridis')
        
        ax.set_xlabel('Strike Price (K)')
        ax.set_ylabel('Time to Maturity (T)')
        ax.set_zlabel('Implied Volatility')
        ax.set_title(f'Heston Model Implied Volatility Surface')
        plt.show()

    def check_feller_condition(self,kappa,theta,sigma):
        """
        Check if the Heston model parameters satisfy the Feller condition:
        2 * kappa * theta >= sigma^2
        """
        feller_condition = 2 * kappa * theta >= sigma**2
        
        if feller_condition:
            print("The Feller condition is satisfied.")
            return True
        else:
            print("The Feller condition is NOT satisfied.")
            return False
   
    def calculate_greeks(self):
        """
        Computes the Greeks for the Heston model using finite differences.
        """
        eps = 1e-6

        delta = (Heston_pricer.FI_for_Greeks(self.payoff,self.S0+eps,self.K,self.T,self.v0,self.r,self.theta,self.sigma,self.kappa,self.rho)-
                 Heston_pricer.FI_for_Greeks(self.payoff,self.S0-eps,self.K,self.T,self.v0,self.r,self.theta,self.sigma,self.kappa,self.rho)) / (2 * eps)
        
        gamma = (Heston_pricer.FI_for_Greeks(self.payoff,self.S0+eps,self.K,self.T,self.v0,self.r,self.theta,self.sigma,self.kappa,self.rho) - 
                 2 * Heston_pricer.FI_for_Greeks(self.payoff,self.S0,self.K,self.T,self.v0,self.r,self.theta,self.sigma,self.kappa,self.rho) +
                 Heston_pricer.FI_for_Greeks(self.payoff,self.S0-eps,self.K,self.T,self.v0,self.r,self.theta,self.sigma,self.kappa,self.rho)) / (eps ** 2)
        
        vega = (Heston_pricer.FI_for_Greeks(self.payoff,self.S0,self.K,self.T,self.v0+eps,self.r,self.theta,self.sigma,self.kappa,self.rho)-
                 Heston_pricer.FI_for_Greeks(self.payoff,self.S0,self.K,self.T,self.v0-eps,self.r,self.theta,self.sigma,self.kappa,self.rho)) / (2 * eps)
        
        rho = (Heston_pricer.FI_for_Greeks(self.payoff,self.S0,self.K,self.T,self.v0,self.r+eps,self.theta,self.sigma,self.kappa,self.rho)-
                 Heston_pricer.FI_for_Greeks(self.payoff,self.S0,self.K,self.T,self.v0,self.r-eps,self.theta,self.sigma,self.kappa,self.rho)) / (2 * eps)

        theta = (Heston_pricer.FI_for_Greeks(self.payoff,self.S0,self.K,self.T,self.v0,self.r,self.theta-eps,self.sigma,self.kappa,self.rho)-
                 Heston_pricer.FI_for_Greeks(self.payoff,self.S0,self.K,self.T,self.v0,self.r,self.theta+eps,self.sigma,self.kappa,self.rho)) / (2 * eps)
        
        # Create DataFrame to display Greeks
        data = {
            "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
            "Value": [delta, gamma, vega, theta, rho],
        }

        return pd.DataFrame(data)

class HestonMCPricer:
    """
    Computes the price of a call or put using the heston pricer
    """
    def __init__(self):
        pass


    def Price(self, S0, v0, K, T, r, kappa, theta, sigma, rho, M, N, option_type='call', visualPaths=False, visualDist=False):
        dt = T / N
        mu = np.array([0, 0])
        cov = np.array([[1, rho], [rho, 1]])

        # Arrays for storing simulated prices and variances
        S = np.full((N + 1, M), S0, dtype=np.float64)
        v = np.full((N + 1, M), v0, dtype=np.float64)

        # Sampling correlated Brownian motions under the risk-neutral measure
        Z = np.random.multivariate_normal(mu, cov, (N, M))

        for i in range(1, N + 1):
            S[i] = S[i - 1] * np.exp((r - 0.5 * v[i - 1]) * dt + np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 0])
            v[i] = np.maximum(v[i - 1] + kappa * (theta - v[i - 1]) * dt + sigma * np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 1], 0)

        if option_type == 'call':
            option_payoff = np.maximum(S[-1] - K, 0)
        elif option_type == 'put':
            option_payoff = np.maximum(K - S[-1], 0)
        else:
            raise ValueError("Invalid option_type. Choose either 'call' or 'put'.")

        # Discounted expected payoff
        option_price = np.exp(-r * T) * np.mean(option_payoff)
        sig = np.std(option_payoff)
        SE = sig / np.sqrt(M)
        if visualPaths:
            plt.figure(figsize=(12, 6)) 
            plt.subplot(1, 2, 1)
            
            plt.plot(S[:, :10])  # Show first 10 paths
            plt.title('Heston Model Price Paths')
            plt.xlabel('Time Steps')
            plt.ylabel('Asset Price')
            
            plt.subplot(1, 2, 2)
            plt.plot(v[:, :10])  # Show first 10 paths
            plt.title('Heston Model Variance Process')
            plt.xlabel('Time Steps')
            plt.ylabel('Variance')
            plt.tight_layout()
            plt.show()

        if visualDist:
            gbm = S0*np.exp( (r - theta**2/2)*T + np.sqrt(theta)*np.sqrt(T)*np.random.normal(0,1,M) )
            fig, ax = plt.subplots()
            ax = sns.kdeplot(S[-1], label=f"rho={rho}", ax=ax)
            ax = sns.kdeplot(gbm, label="GBM", ax=ax)

            plt.title(r'Asset Price Density under Heston Model')
            plt.xlim([20, 180])
            plt.xlabel('$S_T$')
            plt.ylabel('Density')
            plt.legend()
            plt.show()
        return print(f"Option price is {option_price} +/- {SE}.")

class Heston_process:
    """
    Class for the Heston process:
    r = risk free constant rate
    rho = correlation between stock noise and variance noise
    theta = long term mean of the variance process
    sigma = volatility coefficient of the variance process
    kappa = mean reversion coefficient for the variance process
    """

    def __init__(self, mu=0.1, rho=0, sigma=0.2, theta=-0.1, kappa=0.1):
        self.mu = mu
        if np.abs(rho) > 1:
            raise ValueError("|rho| must be <=1")
        self.rho = rho
        if theta < 0 or sigma < 0 or kappa < 0:
            raise ValueError("sigma,theta,kappa must be positive")
        else:
            self.theta = theta
            self.sigma = sigma
            self.kappa = kappa

    def path(self, S0, v0, N, T=1):
        """
        Produces one path of the Heston process.
        N = number of time steps
        T = Time in years
        Returns two arrays S (price) and v (variance).
        """

        MU = np.array([0, 0])
        COV = np.matrix([[1, self.rho], [self.rho, 1]])
        W = ss.multivariate_normal.rvs(mean=MU, cov=COV, size=N - 1)
        W_S = W[:, 0]  # Stock Brownian motion:     W_1
        W_v = W[:, 1]  # Variance Brownian motion:  W_2

        # Initialize vectors
        T_vec, dt = np.linspace(0, T, N, retstep=True)
        dt_sq = np.sqrt(dt)

        X0 = np.log(S0)
        v = np.zeros(N)
        v[0] = v0
        X = np.zeros(N)
        X[0] = X0

        # Generate paths
        for t in range(0, N - 1):
            v_sq = np.sqrt(v[t])
            v[t + 1] = np.abs(v[t] + self.kappa * (self.theta - v[t]) * dt + self.sigma * v_sq * dt_sq * W_v[t])
            X[t + 1] = X[t] + (self.mu - 0.5 * v[t]) * dt + v_sq * dt_sq * W_S[t]

        return np.exp(X), v


class Option_param:
        def __init__(self, S0, K, T, v0, exercise="European", payoff="call"):
            self.S0 = S0  # Initial stock price
            self.K = K  # Strike price
            self.T = T  # Time to maturity
            self.v0 = v0  # Initial variance
                        
            if exercise == "European" or exercise == "American":
                self.exercise = exercise
            else:
                raise ValueError("invalid type. Set 'European' or 'American'")

            if payoff == "call" or payoff == "put":
                self.payoff = payoff
            else:
                raise ValueError("invalid type. Set 'call' or 'put'")

class process_info:
    def __init__(self, mu, sigma, theta, kappa, rho):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.kappa = kappa
        self.rho = rho
        


if __name__ == "__main__":

    strikes = [60, 140]
    maturities = [0.1, 3.0]

    S0 = 100      # Initial stock price
    K = 110       # Strike price
    T = 1.0       # Time to maturity (1 year)
    v0 = 0.05     # Initial variance
    r = 0.05      # Risk-free rate
    sigma = 0.3   # Volatility of variance
    theta = 0.05  # Long-run variance
    kappa = 0.2   # Mean reversion speed
    rho = -0.7    # Correlation between asset and variance (set to negative to capture leverage effect)
    
    N=1000
    M=100000
    S_max = 200    # Maximum stock price
    v_max = 0.6    # Maximum volatility
    # M = 100        # Stock price steps
    # N = 1000       # Variance steps
    # P = 100        # Time steps

    # Test MC pricing
    # pricerMC=HestonMCPricer()
    # pricerMC.Price(S0,v0,K,T,r,kappa,theta, sigma,rho,M,N,option_type='call',visualPaths=False)
   
    # Create option and process information
    option_info = Option_param(S0, K, T, v0,payoff="call")
    HestonProcess = process_info(r, sigma, theta, kappa, rho)

    # Instantiate the Heston pricer
    pricer = Heston_pricer(option_info, HestonProcess)


    # Test Fourier inversion pricing
    start = time.time()
    fi_price = pricer.Fourier_inversion()
    end = time.time()
    print(f"Fourier Inversion Price: {fi_price:.4f}")
    print(f"FI Execution Time: {end - start:.4f} seconds")

    # Test FFT pricing
    FFT_strikes = [110,115,120,125]
    start = time.time()
    fft_prices = pricer.FFT(FFT_strikes)
    end = time.time()
    print(f"FFT Prices: {fft_prices}")
    print(f"FFT Execution Time: {end - start:.4f} seconds")
    print(f"FFT Effective Execution Time: {(end - start)/len(FFT_strikes)} seconds")

    # print(pricer.MC(N,M,True,True))

    strikes = [70, 160]  
    maturities = [0.1, 3.0]

    # print(pricer.calculate_greeks())

    

    # pricer.priceSurface(strikes,maturities)
    pricer.plot_IV_surface(strikes,maturities)
   
                
