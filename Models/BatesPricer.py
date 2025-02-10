import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
import time 
import pandas as pd
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter
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
    
    def Fourier_inversion_manual(self,S0,K,T,r,v0,theta,kappa,sigma,rho,lam,muJ,delta,payoff):
        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(K / S0)  # log moneyness
        cf_H_b_good = partial(
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
        
        return IV_from_Lewis(self.K, self.S0, self.T, self.r, cf_funct)
        
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
                price_surface[j, i] = self.FFT(self.K)
        
        # Plotting the surface
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(K_grid, T_grid, price_surface, cmap='viridis')
        
        ax.set_xlabel('Strike Price (K)')
        ax.set_ylabel('Time to Maturity (T)')
        ax.set_zlabel('Option Price')
        ax.set_title(f'Bates Model {self.payoff.capitalize()} Option Price Surface')
        
        # plt.show()
        return fig
    
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
                iv_value = self.IV_Lewis()
                price_surface[j, i] = np.nan if iv_value == -1 or iv_value < 0 else iv_value
            
        # Smoothing: for each interior cell (i.e. not on an edge),
        # if the IV value is too far from the average of its nonzero neighbors, then replace it.
        tol = 0.001  # tolerance threshold; adjust this value as needed
        for j in range(1, num_T - 1):
            for i in range(1, num_K - 1):
                cell_val = price_surface[j, i]
                # Get immediate neighbors: up, down, left, right.
                neighbors = [
                    price_surface[j-1, i],
                    price_surface[j+1, i],
                    price_surface[j, i-1],
                    price_surface[j, i+1]
                ]
                # Filter out neighbors that are zero or NaN.
                valid_neighbors = [v for v in neighbors if v != 0 and not np.isnan(v)]
                if valid_neighbors:
                    neighbor_avg = np.mean(valid_neighbors)
                    # If the difference is too large, replace the cell value.
                    if abs(cell_val - neighbor_avg) > tol:
                        price_surface[j, i] = neighbor_avg

            # global smoothing
            price_surface = gaussian_filter(price_surface, sigma=0.75)
        
        valid_vals = price_surface[~np.isnan(price_surface)]
        
        if valid_vals.size > 0:
            zmin = np.percentile(valid_vals, 0.05)  # 5th percentile (ignores extreme low values)
            zmax = np.percentile(valid_vals, 99.95)  # 95th percentile (ignores outliers)
            
         
        # Create the 3D plot.
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(K_grid, T_grid, price_surface, cmap='viridis')
        
        ax.set_xlabel('Strike Price (K)')
        ax.set_ylabel('Time to Maturity (T)')
        ax.set_zlabel('Implied Volatility')
        ax.set_title('Bates Model Implied Volatility Surface')
        ax.set_zlim(zmin, zmax)
       
    
        # plt.show()
        return fig

    def priceColorGrid(self, S_range, sigma_range,num_S=15,num_sigma=15, show_pl=False):
        """
        Generate a discrete color grid of European option prices or P/L using the Bates model, with values displayed on each cell.
        """
        S_values = np.linspace(S_range[0], S_range[1], num_S)
        sigma_values = np.linspace(sigma_range[0], sigma_range[1], num_sigma)

        # Create a meshgrid for spot prices and volatilities
        S_grid, sigma_grid = np.meshgrid(S_values, sigma_values)

        # Compute option prices or P/L for each combination of S and sigma
        values = np.zeros_like(S_grid, dtype=float)
        for i in range(S_grid.shape[0]):
            for j in range(S_grid.shape[1]):
                S = S_grid[i, j]
                nu = sigma_grid[i, j]
                if show_pl:
                    # Compute P/L as the difference between price and intrinsic value
                    intrinsic_value = max(0, (S - self.K) if self.payoff == 'call' else (self.K - S))
                    values[i, j] = self.Fourier_inversion_manual(
                        S,
                        self.K,
                        self.T,
                        self.r,
                        nu,
                        self.theta,
                        self.kappa,
                        self.sigma,
                        self.rho,
                        self.lam,
                        self.muJ,
                        self.delta,
                        self.payoff
                    ) - intrinsic_value
                else:
                    # Get the price
                    values[i, j] = self.Fourier_inversion_manual(
                        S,
                        self.K,
                        self.T,
                        self.r,
                        nu,
                        self.theta,
                        self.kappa,
                        self.sigma,
                        self.rho,
                        self.lam,
                        self.muJ,
                        self.delta,
                        self.payoff
                    )

        # # Plot the 2D discrete color grid
        # plt.figure(figsize=(12, 8))
        # cmap = plt.get_cmap('RdYlGn' if show_pl else 'viridis')  
        # plt.pcolormesh(S_grid, sigma_grid, values, cmap=cmap, shading='auto', edgecolors='k', linewidth=0.5)
        # plt.colorbar(label='Option Price' if not show_pl else 'P/L')

        # # Annotate each cell with the value
        # for i in range(S_grid.shape[0]):
        #     for j in range(S_grid.shape[1]):
        #         plt.text(S_grid[i, j], sigma_grid[i, j], f'{values[i, j]:.2f}', ha='center', va='center', fontsize=8, color='black')

        # # Add labels and title
        # plt.xlabel('Spot Price (S)')
        # plt.ylabel('Implied Volatility (sigma)')
        # plt.title(f"Option {'P/L' if show_pl else 'Price'} Color Grid ({self.payoff.capitalize()} Option)")
        # plt.show()
        
        fig = Figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        cmap = plt.get_cmap('RdYlGn' if show_pl else 'viridis')
        pcm = ax.pcolormesh(S_grid, sigma_grid, values, cmap=cmap,
                            shading='auto', edgecolors='k', linewidth=0.5)
        
        fig.colorbar(pcm, ax=ax, label='Option Price' if not show_pl else 'P/L')
        
        for i in range(S_grid.shape[0]):
            for j in range(S_grid.shape[1]):
                ax.text(S_grid[i, j], sigma_grid[i, j], f'{values[i, j]:.2f}',
                        ha='center', va='center', fontsize=8, color='black')
        
        ax.set_xlabel('Spot Price (S)')
        ax.set_ylabel('Implied Instantanous Volatility (v0)')
        ax.set_title(f"Option {'P/L' if show_pl else 'Price'} Color Grid ({self.payoff.capitalize()} Option)")
        
        return fig
    
    def plot_stockpaths(self,N=1000, M=10):
        dt = self.T / N
        mu_vec = np.array([0, 0])
        cov = np.array([[1, self.rho], [self.rho, 1]])

        # Preallocate arrays for prices and variances.
        S = np.full((N + 1, M), self.S0, dtype=np.float64)
        v = np.full((N + 1, M), self.v0, dtype=np.float64)

        # Sample correlated Brownian increments for the Heston dynamics.
        Z = np.random.multivariate_normal(mu_vec, cov, (N, M))

        for i in range(1, N + 1):
            # Diffusion part for the asset price.
            diffusion = np.exp((self.r - 0.5 * v[i - 1]) * dt + np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 0])
            
            # Jump multiplier: if lam > 0, simulate jumps.
            jump_multiplier = np.ones(M)
            if self.lam > 0:
                n_jumps = np.random.poisson(self.lam * dt, M)
                Z_jump = np.random.normal(0, 1, M)
                jump_size = np.where(n_jumps > 0,
                                     n_jumps * self.muJ + np.sqrt(n_jumps) * self.delta * Z_jump,
                                     0.0)
                jump_multiplier = np.exp(jump_size)
            
            # Update asset price.
            S[i] = S[i - 1] * diffusion * jump_multiplier

            # Update variance process (ensure non-negativity).
            v[i] = np.maximum(
                v[i - 1] + self.kappa * (self.theta - v[i - 1]) * dt + self.sigma * np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 1],
                0
            )

        fig_paths = plt.figure(figsize=(12, 6))
        # Left subplot: asset price paths
        ax1 = fig_paths.add_subplot(1, 2, 1)
        ax1.plot(S[:, :M])
        ax1.set_title(f'Bates Model Price Paths ({M} paths)')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Asset Price')
        # Right subplot: variance paths
        ax2 = fig_paths.add_subplot(1, 2, 2)
        ax2.plot(v[:, :M])
        ax2.set_title(f'Bates Variance Process ({M} paths)')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Variance')
        fig_paths.tight_layout()
        return fig_paths

    def plot_dist(self,N=1000,M=10000):
        dt = self.T / N
        mu_vec = np.array([0, 0])
        cov = np.array([[1, self.rho], [self.rho, 1]])

        # Preallocate arrays for prices and variances.
        S = np.full((N + 1, M), self.S0, dtype=np.float64)
        v = np.full((N + 1, M), self.v0, dtype=np.float64)

        # Sample correlated Brownian increments for the Heston dynamics.
        Z = np.random.multivariate_normal(mu_vec, cov, (N, M))

        for i in range(1, N + 1):
            # Diffusion part for the asset price.
            diffusion = np.exp((self.r - 0.5 * v[i - 1]) * dt + np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 0])
            
            # Jump multiplier: if lam > 0, simulate jumps.
            jump_multiplier = np.ones(M)
            if self.lam > 0:
                n_jumps = np.random.poisson(self.lam * dt, M)
                Z_jump = np.random.normal(0, 1, M)
                jump_size = np.where(n_jumps > 0,
                                     n_jumps * self.muJ + np.sqrt(n_jumps) * self.delta * Z_jump,
                                     0.0)
                jump_multiplier = np.exp(jump_size)
            
            # Update asset price.
            S[i] = S[i - 1] * diffusion * jump_multiplier

            # Update variance process (ensure non-negativity).
            v[i] = np.maximum(
                v[i - 1] + self.kappa * (self.theta - v[i - 1]) * dt + self.sigma * np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 1],
                0
            )
        gbm = self.S0 * np.exp((self.r - self.theta**2 / 2) * self.T + np.sqrt(self.theta * self.T) * np.random.normal(0, 1, M))
        fig_dist, ax = plt.subplots(figsize=(8, 5))
        # Use seaborn's kdeplot to plot densities.
        sns.kdeplot(S[-1], label=f"rho={self.rho}", ax=ax)
        sns.kdeplot(gbm, label="GBM", ax=ax)
        ax.set_title('Asset Price Density under Bates Model')
        ax.set_xlabel('$S_T$')
        ax.set_ylabel('Density')
        ax.legend()
        return fig_dist

    def MC(self, M, N):
        """
        Simulate the Bates model using Monte Carlo.

        Parameters:
            M       : number of simulation paths.
            N       : number of time steps.
        Returns:
            Prints the option price and standard error.
        """
        start = time.time()
        dt = self.T / N
        mu_vec = np.array([0, 0])
        cov = np.array([[1, self.rho], [self.rho, 1]])

        # Preallocate arrays for prices and variances.
        S = np.full((N + 1, M), self.S0, dtype=np.float64)
        v = np.full((N + 1, M), self.v0, dtype=np.float64)

        # Sample correlated Brownian increments for the Heston dynamics.
        Z = np.random.multivariate_normal(mu_vec, cov, (N, M))

        for i in range(1, N + 1):
            # Simulate the asset price diffusion part
            diffusion = np.exp((self.r - 0.5 * v[i - 1]) * dt + np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 0])
            
            # Initialize jump multiplier as 1 (no jump)
            jump_multiplier = np.ones(M)
            if self.lam > 0:
                # For each path, sample the number of jumps in this time step
                n_jumps = np.random.poisson(self.lam * dt, M)
                # For paths with jumps, simulate an aggregated jump size
                # aggregated jump = n * muJ + sqrt(n)*delta * Z_2, where Z_2 ~ N(0,1)
                # Note: if n_jumps is 0 then the term will be zero.
                # We use np.where to avoid taking sqrt of 0 (or negative) n_jumps.
                Z_jump = np.random.normal(0, 1, M)
                jump_size = np.where(n_jumps > 0,
                                     n_jumps * self.muJ + np.sqrt(n_jumps) * self.delta * Z_jump,
                                     0.0)
                jump_multiplier = np.exp(jump_size)
            
            # Update asset price including both diffusion and jumps.
            S[i] = S[i - 1] * diffusion * jump_multiplier

            # Update variance process (ensuring non-negativity)
            v[i] = np.maximum(
                v[i - 1] + self.kappa * (self.theta - v[i - 1]) * dt + self.sigma * np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 1],
                0
            )

        # Compute option payoff at maturity.
        if self.payoff == 'call':
            option_payoff = np.maximum(S[-1] - self.K, 0)
        elif self.payoff == 'put':
            option_payoff = np.maximum(self.K - S[-1], 0)
        else:
            raise ValueError("Invalid option type. Choose either 'call' or 'put'.")

        # Discounted expected payoff.
        option_price = np.exp(-self.r * self.T) * np.mean(option_payoff)
        sig = np.std(option_payoff)
        SE = sig / np.sqrt(M)

        end = time.time()
        comptime = end - start
        return option_price, SE, comptime
    
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
    v0 = 0.5     # Initial variance
    r = 0.05      # Risk-free rate
    sigma = 0.3   # Volatility of variance
    theta = 0.2  # Long-run variance
    kappa = 1   # Mean reversion speed
    rho = -0.7    # Correlation between asset and variance (set to negative to capture leverage effect)
    lam = 1.0
    muJ = 0.3
    delta = 0.2


    N=1000
    M=100000
    S_max = 200    # Maximum stock price
    v_max = 0.6    # Maximum volatility
    M = 1000        # Stock price steps
    N = 1000         # Variance steps
    P = 1000        # Time steps

    # Test MC pricing
    # pricerMC=BatesMCPricer()
    # pricerMC.Price(S0,v0,K,T,r,kappa,theta, sigma,rho,M,N,option_type='call',lam=lam,muJ=muJ,delta=delta,visualPaths=True,visualDist=True)
   
    # Create option and process information
    option_info = Option_param(S0, K, T, v0,payoff="call")
    Bates_process = process_info(r, sigma, theta, kappa, rho,lam,muJ,delta)

    # Instantiate the Heston pricer
    pricer = Bates_pricer(option_info, Bates_process)
    

                
