import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from functools import partial
import time 
import pandas as pd
from prereq import fft_Lewis, IV_from_Lewis, Q1, Q2
from HestonModelPricer import cf_Heston_good

def cf_Bates(u, t, v0, mu, kappa, theta, sigma, rho, lam, muJ, delta):
    """
    Bates model characteristic function:
    Combines the Heston part with an exponential jump component.
    """
    mu_adj = mu - lam * (np.exp(muJ + 0.5 * delta**2) - 1)
    cf_heston = cf_Heston_good(u, t, v0, mu_adj, kappa, theta, sigma, rho)
    cf_jump = np.exp(lam * t * (np.exp(1j * u * muJ - 0.5 * delta**2 * u**2) - 1))
    return cf_heston * cf_jump




class Bates_pricer:
    """
    Class to price the options with the Bates model by:
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
        self.sigma = Process_info.sigma  # Bates parameter
        self.theta = Process_info.theta  # Bates parameter
        self.kappa = Process_info.kappa  # Bates parameter
        self.rho = Process_info.rho  # Bates parameter

        self.lam = Process_info.lam
        self.muJ = Process_info.muJ
        self.delta = Process_info.delta

        self.S0 = Option_info.S0  # current price
        self.v0 = Option_info.v0  # spot variance
        self.K = Option_info.K  # strike
        self.T = Option_info.T  # maturity in years

        self.exercise = Option_info.exercise
        self.payoff = Option_info.payoff
    
    
    def FI_for_Greeks(payoff, S0, K,T,v0,r,theta,sigma,kappa,rho,lam,muJ,delta):
        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(K / S0)  # log moneyness
        cf_funct = partial(
            cf_Bates,
            t=T,
            v0=v0,
            mu=r,
            theta=theta,
            sigma=sigma,
            kappa=kappa,
            rho=rho,
            lam=lam,
            muJ=muJ,
            delta=delta,
        )

        limit_max = 2000  # right limit in the integration

        if payoff == "call":
            call = S0 * Q1(k, cf_funct, limit_max) - K * np.exp(-r * T) * Q2(
                k, cf_funct, limit_max
            )
            return call
        elif payoff == "put":
            put = K * np.exp(-r * T) * (1 - Q2(k, cf_funct, limit_max)) - S0 * (
                1 - Q1(k, cf_funct, limit_max)
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
            cf_Bates,
            t=self.T,
            v0=self.v0,
            mu=self.r,
            theta=self.theta,
            sigma=self.sigma,
            kappa=self.kappa,
            rho=self.rho,
            lam=self.lam,
            muJ=self.muJ,
            delta=self.delta,
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
        cf_funct = partial(
            cf_Bates,
            t=self.T,
            v0=self.v0,
            mu=self.r,
            theta=self.theta,
            sigma=self.sigma,
            kappa=self.kappa,
            rho=self.rho,
            lam=self.lam,
            muJ=self.muJ,
            delta=self.delta,
        )

        if self.payoff == "call":
            return fft_Lewis(K, self.S0, self.r, self.T, cf_funct, interp="cubic")
        elif self.payoff == "put":  # put-call parity
            return (
                fft_Lewis(K, self.S0, self.r, self.T, cf_funct, interp="cubic")
                - self.S0
                + K * np.exp(-self.r * self.T)
            )
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def IV_Lewis(self):
        """Implied Volatility from the Lewis formula"""

        cf_funct = partial(
            cf_Bates,
            t=self.T,
            v0=self.v0,
            mu=self.r,
            theta=self.theta,
            sigma=self.sigma,
            kappa=self.kappa,
            rho=self.rho,
            lam=self.lam,
            muJ=self.muJ,
            delta=self.delta,
        )
        if self.payoff == "call":
            return IV_from_Lewis(self.K, self.S0, self.T, self.r, cf_funct)
        elif self.payoff == "put":
            raise NotImplementedError
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")
        
    def priceSurface(self, K_range, T_range, num_K=10, num_T=10):
        """
        Generate a 3D surface plot for option prices under the Bates model.
        
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
        ax.set_title(f'Bates Model {self.payoff.capitalize()} Option Price Surface')
        
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
        ax.set_title(f'Bates Model Implied Volatility Surface')
        plt.show()

    def check_feller_condition(self,kappa,theta,sigma):
        """
        Check if the Bates model parameters satisfy the Feller condition:
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
        Computes the Greeks for the Bates model using finite differences.
        """
        eps = 1e-6

        delta = (Bates_pricer.FI_for_Greeks(self.payoff,self.S0+eps,self.K,self.T,self.v0,self.r,self.theta,self.sigma,self.kappa,self.rho,self.lam,self.muJ,self.delta)-
                 Bates_pricer.FI_for_Greeks(self.payoff,self.S0-eps,self.K,self.T,self.v0,self.r,self.theta,self.sigma,self.kappa,self.rho,self.lam,self.muJ,self.delta)) / (2 * eps)
        
        gamma = (Bates_pricer.FI_for_Greeks(self.payoff,self.S0+eps,self.K,self.T,self.v0,self.r,self.theta,self.sigma,self.kappa,self.rho,self.lam,self.muJ,self.delta) - 
                 2 * Bates_pricer.FI_for_Greeks(self.payoff,self.S0,self.K,self.T,self.v0,self.r,self.theta,self.sigma,self.kappa,self.rho,self.lam,self.muJ,self.delta) +
                 Bates_pricer.FI_for_Greeks(self.payoff,self.S0-eps,self.K,self.T,self.v0,self.r,self.theta,self.sigma,self.kappa,self.rho,self.lam,self.muJ,self.delta)) / (eps ** 2)
        
        vega = (Bates_pricer.FI_for_Greeks(self.payoff,self.S0,self.K,self.T,self.v0+eps,self.r,self.theta,self.sigma,self.kappa,self.rho,self.lam,self.muJ,self.delta)-
                 Bates_pricer.FI_for_Greeks(self.payoff,self.S0,self.K,self.T,self.v0-eps,self.r,self.theta,self.sigma,self.kappa,self.rho,self.lam,self.muJ,self.delta)) / (2 * eps)
        
        rho = (Bates_pricer.FI_for_Greeks(self.payoff,self.S0,self.K,self.T,self.v0,self.r+eps,self.theta,self.sigma,self.kappa,self.rho,self.lam,self.muJ,self.delta)-
                 Bates_pricer.FI_for_Greeks(self.payoff,self.S0,self.K,self.T,self.v0,self.r-eps,self.theta,self.sigma,self.kappa,self.rho,self.lam,self.muJ,self.delta)) / (2 * eps)

        theta = (Bates_pricer.FI_for_Greeks(self.payoff,self.S0,self.K,self.T-eps,self.v0,self.r,self.theta,self.sigma,self.kappa,self.rho,self.lam,self.muJ,self.delta)-
                 Bates_pricer.FI_for_Greeks(self.payoff,self.S0,self.K,self.T+eps,self.v0,self.r,self.theta,self.sigma,self.kappa,self.rho,self.lam,self.muJ,self.delta)) / (2 * eps)
        
        # Create DataFrame to display Greeks
        data = {
            "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
            "Value": [delta, gamma, vega, theta, rho],
        }
        df = pd.DataFrame(data).set_index("Greek")
        return df

class BatesMCPricer:
    """
    Computes the price of a call or put using Monte Carlo simulation under the Bates model,
    i.e., the Heston stochastic volatility model with jumps.
    """
    def __init__(self):
        pass

    def Price(
        self,
        S0, v0, K, T, r, kappa, theta, sigma, rho,
        M, N,
        option_type='call',
        lam=0.0,      # jump intensity
        muJ=0.0,      # mean jump size
        delta=0.0,    # jump volatility
        visualPaths=False,
        visualDist=False
    ):
        """
        Simulate the Bates model using Monte Carlo.

        Parameters:
            S0      : initial asset price.
            v0      : initial variance.
            K       : strike.
            T       : time to maturity.
            r       : risk-free rate.
            kappa   : Heston mean-reversion speed.
            theta   : long-term variance.
            sigma   : vol-of-vol.
            rho     : correlation between the asset and variance Brownian motions.
            M       : number of simulation paths.
            N       : number of time steps.
            option_type: 'call' or 'put'.
            lam     : jump intensity (λ). If lam==0, then no jumps occur.
            muJ     : mean jump size.
            delta   : jump volatility.
            visualPaths: if True, plot sample paths.
            visualDist: if True, plot the distribution of simulated terminal prices.
        Returns:
            Prints the option price and standard error.
        """
        dt = T / N
        mu_vec = np.array([0, 0])
        cov = np.array([[1, rho], [rho, 1]])

        # Preallocate arrays for prices and variances.
        S = np.full((N + 1, M), S0, dtype=np.float64)
        v = np.full((N + 1, M), v0, dtype=np.float64)

        # Sample correlated Brownian increments for the Heston dynamics.
        Z = np.random.multivariate_normal(mu_vec, cov, (N, M))

        for i in range(1, N + 1):
            # Simulate the asset price diffusion part
            diffusion = np.exp((r - 0.5 * v[i - 1]) * dt + np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 0])
            
            # Initialize jump multiplier as 1 (no jump)
            jump_multiplier = np.ones(M)
            if lam > 0:
                # For each path, sample the number of jumps in this time step
                n_jumps = np.random.poisson(lam * dt, M)
                # For paths with jumps, simulate an aggregated jump size
                # aggregated jump = n * muJ + sqrt(n)*delta * Z_2, where Z_2 ~ N(0,1)
                # Note: if n_jumps is 0 then the term will be zero.
                # We use np.where to avoid taking sqrt of 0 (or negative) n_jumps.
                Z_jump = np.random.normal(0, 1, M)
                jump_size = np.where(n_jumps > 0,
                                     n_jumps * muJ + np.sqrt(n_jumps) * delta * Z_jump,
                                     0.0)
                jump_multiplier = np.exp(jump_size)
            
            # Update asset price including both diffusion and jumps.
            S[i] = S[i - 1] * diffusion * jump_multiplier

            # Update variance process (ensuring non-negativity)
            v[i] = np.maximum(
                v[i - 1] + kappa * (theta - v[i - 1]) * dt + sigma * np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 1],
                0
            )

        # Compute option payoff at maturity.
        if option_type == 'call':
            option_payoff = np.maximum(S[-1] - K, 0)
        elif option_type == 'put':
            option_payoff = np.maximum(K - S[-1], 0)
        else:
            raise ValueError("Invalid option_type. Choose either 'call' or 'put'.")

        # Discounted expected payoff.
        option_price = np.exp(-r * T) * np.mean(option_payoff)
        sig = np.std(option_payoff)
        SE = sig / np.sqrt(M)

        if visualPaths:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(S[:, :10])
            plt.title('Bates Model Price Paths (10 paths)')
            plt.xlabel('Time Steps')
            plt.ylabel('Asset Price')

            plt.subplot(1, 2, 2)
            plt.plot(v[:, :10])
            plt.title('Heston Variance Process (10 paths)')
            plt.xlabel('Time Steps')
            plt.ylabel('Variance')
            plt.tight_layout()
            plt.show()

        if visualDist:
            # For comparison, simulate GBM with the same drift and volatility.
            gbm = S0 * np.exp((r - theta**2/2) * T + np.sqrt(theta * T) * np.random.normal(0, 1, M))
            fig, ax = plt.subplots()
            sns.kdeplot(S[-1], label=f"rho={rho}", ax=ax)
            sns.kdeplot(gbm, label="GBM", ax=ax)
            plt.title('Asset Price Density under Bates Model')
            plt.xlabel('$S_T$')
            plt.ylabel('Density')
            plt.legend()
            plt.show()

        print(f"Option price is {option_price} +/- {SE}.")
        return option_price

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
    def __init__(self, mu, sigma, theta, kappa, rho, lam=0.0, muJ=0.0, delta=0.0):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.kappa = kappa
        self.rho = rho
        self.lam = lam # Jump intensity      
        self.muJ = muJ # mean jump size      
        self.delta = delta # Jump volatility

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
    lam = 1.0
    muJ = 0.01
    delta = 0.1


    N=1000
    M=100000
    S_max = 200    # Maximum stock price
    v_max = 0.6    # Maximum volatility
    M = 1000        # Stock price steps
    N = 1000         # Variance steps
    P = 1000        # Time steps

    # Test MC pricing
    pricerMC=BatesMCPricer()
    pricerMC.Price(S0,v0,K,T,r,kappa,theta, sigma,rho,M,N,option_type='call',lam=lam,muJ=muJ,delta=delta,visualPaths=True,visualDist=True)
   
    # Create option and process information
    option_info = Option_param(S0, K, T, v0,payoff="call")
    Bates_process = process_info(r, sigma, theta, kappa, rho,lam,muJ,delta)

    # Instantiate the Heston pricer
    pricer = Bates_pricer(option_info, Bates_process)


    # Test Fourier inversion pricing
    start = time.time()
    fi_price = pricer.Fourier_inversion()
    end = time.time()
    print(f"Fourier Inversion Price: {fi_price:.4f}")
    print(f"FI Execution Time: {end - start:.4f} seconds")

    # Test FFT pricing
    start = time.time()
    fft_prices = pricer.FFT(K)
    end = time.time()
    print(f"FFT Prices: {fft_prices:.4f}")
    print(f"FFT Execution Time: {end - start:.4f} seconds")
    strikes = [70, 160]  
    maturities = [0.1, 3.0]

    # print(pricer.calculate_greeks())
    mcpricer = BatesMCPricer()
    mcpricer.Price(S0,v0,K,T,r,kappa,theta,sigma,rho,M,N,'call',lam,muJ,delta,True,True)
    

    pricer.priceSurface(strikes,maturities)
    pricer.plot_IV_surface(strikes,maturities)
   
                
