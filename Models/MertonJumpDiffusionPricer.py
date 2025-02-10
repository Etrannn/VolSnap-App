from scipy import sparse
from scipy.sparse.linalg import splu
from time import time
import numpy as np
import scipy.stats as ss
from scipy import signal
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from math import factorial
from functools import partial
from blackScholesPricer import BS_pricer
from scipy.stats import norm
import pandas as pd
from scipy.ndimage import gaussian_filter
from prereq import fft_Lewis, IV_from_Lewis, Q1, Q2, Gil_Pelaez_pdf, Merton_pdf



class MJDmonteCarlo:

    def __init__(self):
        pass

    def merton_jump_paths(self,S, T, r, sigma,  lam, m, v, steps, Npaths):
        size=(steps,Npaths)
        dt = T/steps 
        poi_rv = np.multiply(np.random.poisson( lam*dt, size=size),
                            np.random.normal(m,v, size=size)).cumsum(axis=0)
        geo = np.cumsum(((r -  sigma**2/2 -lam*(m  + v**2*0.5))*dt +\
                                sigma*np.sqrt(dt) * \
                                np.random.normal(size=size)), axis=0)
        
        return np.exp(geo+poi_rv)*S

    def price(self,S, K, T, r, sigma,  lam, m, v, steps, Npaths, option_type='call'):
        ST = self.merton_jump_paths(S, T, r, sigma,  lam, m, v, steps, Npaths)
        if option_type == 'call':
            CT = np.maximum(0,ST[-1]-K)
            C0 = np.exp(-r*T)*np.mean(CT)
            sigma = np.std(CT)
            SE = sigma / np.sqrt(Npaths)
            return C0, SE
        
        elif option_type == 'put':
            PT = max(0,K-ST[-1])
            P0 = np.exp(-r*T)*np.mean(PT)
            sigma = np.std(PT)
            SE = sigma / np.sqrt(Npaths)
            return P0, SE
        else:
            raise ValueError("Invalid option type. Must be 'call' or 'put'.")
    
    def visualiseMertonJumpPaths(self, S, T, r, sigma,  lam, m, v, steps, Npaths):
        j = self.merton_jump_paths(S, T, r, sigma,  lam, m, v, steps, Npaths)
        plt.plot(j)
        plt.xlabel('Days')
        plt.ylabel('Stock Price')
        plt.title(f'Jump Diffusion Process showing {Npaths} paths')
        plt.show()

def cf_mert(u, t=1, mu=1, sig=2, lam=0.8, muJ=0, sigJ=0.5):
    """
    Characteristic function of a Merton random variable at time t
    mu: drift
    sig: diffusion coefficient
    lam: jump activity
    muJ: jump mean size
    sigJ: jump size standard deviation
    """
    return np.exp(
        t * (1j * u * mu - 0.5 * u**2 * sig**2 + lam * (np.exp(1j * u * muJ - 0.5 * u**2 * sigJ**2) - 1))
    )

class Merton_pricer:
    """
    Closed Formula.
    Monte Carlo.
    Finite-difference PIDE: Explicit-implicit scheme

        0 = dV/dt + (r -(1/2)sig^2 -m) dV/dx + (1/2)sig^2 d^V/dx^2
                 + \int[ V(x+y) nu(dy) ] -(r+lam)V
    """

    def __init__(self, Option_info, Process_info):
        """
        Process_info:  of type Merton_process. It contains (r, sig, lam, muJ, sigJ) i.e.
        interest rate, diffusion coefficient, jump activity and jump distribution params

        Option_info:  of type Option_param. It contains (S0,K,T) i.e. current price,
        strike, maturity in years
        """
        self.r = Process_info.r  # interest rate
        self.sig = Process_info.sig  # diffusion coefficient
        self.lam = Process_info.lam  # jump activity
        self.muJ = Process_info.muJ  # jump mean
        self.sigJ = Process_info.sigJ  # jump std
        self.exp_RV = Process_info.exp_RV  # function to generate exponential Merton Random Variables

        self.S0 = Option_info.S0  # current price
        self.K = Option_info.K  # strike
        self.T = Option_info.T  # maturity in years
        
        self.Option_info = Option_info
        self.Process_info = Process_info

        self.price = 0
        self.S_vec = None
        self.price_vec = None
        self.mesh = None
        self.exercise = Option_info.exercise
        self.payoff = Option_info.payoff

    def payoff_f(self, S):
        if self.payoff == "call":
            Payoff = np.maximum(S - self.K, 0)
        elif self.payoff == "put":
            Payoff = np.maximum(self.K - S, 0)
        return Payoff
    
    @staticmethod
    def MertonClosedForm(payoff="call", S0=100.0, K=100.0, T=1.0, r=0.1, sig=0.2,muJ=0.5,sigJ=0.3,lam=0.5):
        """
        Merton closed formula with variables.
        """

        m = lam * (np.exp(muJ + (sigJ**2) / 2) - 1)  # coefficient m
        lam2 = lam * np.exp(muJ + (sigJ**2) / 2)

        tot = 0
        for i in range(18):
            tot += (np.exp(-lam2 * T) * (lam2 * T) ** i / factorial(i)) * BS_pricer.BlackScholes(
                payoff,
                S0,
                K,
                T,
                 - m + i * (muJ + 0.5 * sigJ**2) / T,
                np.sqrt(sig**2 + (i * sigJ**2) / T)
                
            )
        return tot
    
    def closed_formula(self):
        """
        Merton closed formula.
        """

        m = self.lam * (np.exp(self.muJ + (self.sigJ**2) / 2) - 1)  # coefficient m
        lam2 = self.lam * np.exp(self.muJ + (self.sigJ**2) / 2)

        tot = 0
        for i in range(18):
            tot += (np.exp(-lam2 * self.T) * (lam2 * self.T) ** i / factorial(i)) * BS_pricer.BlackScholes(
                self.payoff,
                self.S0,
                self.K,
                self.T,
                self.r - m + i * (self.muJ + 0.5 * self.sigJ**2) / self.T,
                np.sqrt(self.sig**2 + (i * self.sigJ**2) / self.T)
                
            )
        return tot
    
    def FourierInvCalib(self, K, T, sigma, lamb, muJ, sigmaJ):
        """
        Price obtained by inversion of the characteristic function used for calibration
        """
        k = np.log(K / self.S0)  # log moneyness
        m = self.lam * (np.exp(self.muJ + (self.sigJ**2) / 2) - 1)  # coefficient m
        cf_Mert = partial(
            cf_mert,
            t=T,
            mu=(self.r - 0.5 * self.sig**2 - m),
            sig=sigma,
            lam=lamb,
            muJ=muJ,
            sigJ=sigmaJ,
        )

        if self.payoff == "call":
            call = self.S0 * Q1(k, cf_Mert, np.inf) - self.K * np.exp(-self.r * self.T) * Q2(
                k, cf_Mert, np.inf
            )  # pricing function
            return call
        elif self.payoff == "put":
            put = self.K * np.exp(-self.r * self.T) * (1 - Q2(k, cf_Mert, np.inf)) - self.S0 * (
                1 - Q1(k, cf_Mert, np.inf)
            )  # pricing function
            return put
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")
        
    def Fourier_inversion(self):
        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(self.K / self.S0)  # log moneyness
        m = self.lam * (np.exp(self.muJ + (self.sigJ**2) / 2) - 1)  # coefficient m
        cf_Mert = partial(
            cf_mert,
            t=self.T,
            mu=(self.r - 0.5 * self.sig**2 - m),
            sig=self.sig,
            lam=self.lam,
            muJ=self.muJ,
            sigJ=self.sigJ,
        )

        if self.payoff == "call":
            call = self.S0 * Q1(k, cf_Mert, np.inf) - self.K * np.exp(-self.r * self.T) * Q2(
                k, cf_Mert, np.inf
            )  # pricing function
            return call
        elif self.payoff == "put":
            put = self.K * np.exp(-self.r * self.T) * (1 - Q2(k, cf_Mert, np.inf)) - self.S0 * (
                1 - Q1(k, cf_Mert, np.inf)
            )  # pricing function
            return put
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def FFT(self, K):
        """
        FFT method. It returns a vector of prices.
        K is an array of strikes
        """
        K = np.array(K)
        m = self.lam * (np.exp(self.muJ + (self.sigJ**2) / 2) - 1)  # coefficient m
        cf_Mert = partial(
            cf_mert,
            t=self.T,
            mu=(self.r - 0.5 * self.sig**2 - m),
            sig=self.sig,
            lam=self.lam,
            muJ=self.muJ,
            sigJ=self.sigJ,
        )

        if self.payoff == "call":
            return fft_Lewis(K, self.S0, self.r, self.T, cf_Mert, interp="cubic")
        elif self.payoff == "put":  # put-call parity
            return (
                fft_Lewis(K, self.S0, self.r, self.T, cf_Mert, interp="cubic") - self.S0 + K * np.exp(-self.r * self.T)
            )
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def IV_Lewis(self):
        """Implied Volatility from the Lewis formula"""

        m = self.lam * (np.exp(self.muJ + (self.sigJ**2) / 2) - 1)  # coefficient m
        cf_Mert = partial(
            cf_mert,
            t=self.T,
            mu=(self.r - 0.5 * self.sig**2 - m),
            sig=self.sig,
            lam=self.lam,
            muJ=self.muJ,
            sigJ=self.sigJ,
        )

        return IV_from_Lewis(self.K, self.S0, self.T, self.r, cf_Mert)

    def MC(self, N, Err=False, Time=False):
        """
        Merton Monte Carlo
        Err = return Standard Error if True
        Time = return execution time if True
        """
        t_init = time()

        S_T = self.exp_RV(self.S0, self.T, N)
        V = np.mean(np.exp(-self.r * self.T) * self.payoff_f(S_T), axis=0)

        if Err is True:
            if Time is True:
                elapsed = time() - t_init
                return V, ss.sem(np.exp(-self.r * self.T) * self.payoff_f(S_T)), elapsed
            else:
                return V, ss.sem(np.exp(-self.r * self.T) * self.payoff_f(S_T))
        else:
            if Time is True:
                elapsed = time() - t_init
                return V, elapsed
            else:
                return V

    def PIDE_price(self, steps, Time=False):
        """
        steps = tuple with number of space steps and time steps
        payoff = "call" or "put"
        exercise = "European" or "American"
        Time = Boolean. Execution time.
        """
        t_init = time()

        Nspace = steps[0]
        Ntime = steps[1]

        S_max = 6 * float(self.K)
        S_min = float(self.K) / 6
        x_max = np.log(S_max)
        x_min = np.log(S_min)

        dev_X = np.sqrt(self.lam * self.sigJ**2 + self.lam * self.muJ**2)

        dx = (x_max - x_min) / (Nspace - 1)
        extraP = int(np.floor(5 * dev_X / dx))  # extra points beyond the B.C.
        x = np.linspace(x_min - extraP * dx, x_max + extraP * dx, Nspace + 2 * extraP)  # space discretization
        t, dt = np.linspace(0, self.T, Ntime, retstep=True)  # time discretization

        Payoff = self.payoff_f(np.exp(x))
        offset = np.zeros(Nspace - 2)
        V = np.zeros((Nspace + 2 * extraP, Ntime))  # grid initialization

        if self.payoff == "call":
            V[:, -1] = Payoff  # terminal conditions
            V[-extraP - 1 :, :] = np.exp(x[-extraP - 1 :]).reshape(extraP + 1, 1) * np.ones(
                (extraP + 1, Ntime)
            ) - self.K * np.exp(-self.r * t[::-1]) * np.ones(
                (extraP + 1, Ntime)
            )  # boundary condition
            V[: extraP + 1, :] = 0
        else:
            V[:, -1] = Payoff
            V[-extraP - 1 :, :] = 0
            V[: extraP + 1, :] = self.K * np.exp(-self.r * t[::-1]) * np.ones((extraP + 1, Ntime))

        cdf = ss.norm.cdf(
            [np.linspace(-(extraP + 1 + 0.5) * dx, (extraP + 1 + 0.5) * dx, 2 * (extraP + 2))],
            loc=self.muJ,
            scale=self.sigJ,
        )[0]
        nu = self.lam * (cdf[1:] - cdf[:-1])

        lam_appr = sum(nu)
        m_appr = np.array([np.exp(i * dx) - 1 for i in range(-(extraP + 1), extraP + 2)]) @ nu

        sig2 = self.sig**2
        dxx = dx**2
        a = (dt / 2) * ((self.r - m_appr - 0.5 * sig2) / dx - sig2 / dxx)
        b = 1 + dt * (sig2 / dxx + self.r + lam_appr)
        c = -(dt / 2) * ((self.r - m_appr - 0.5 * sig2) / dx + sig2 / dxx)

        D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()
        DD = splu(D)
        if self.exercise == "European":
            for i in range(Ntime - 2, -1, -1):
                offset[0] = a * V[extraP, i]
                offset[-1] = c * V[-1 - extraP, i]
                V_jump = V[extraP + 1 : -extraP - 1, i + 1] + dt * signal.convolve(
                    V[:, i + 1], nu[::-1], mode="valid", method="fft"
                )
                V[extraP + 1 : -extraP - 1, i] = DD.solve(V_jump - offset)
        elif self.exercise == "American":
            for i in range(Ntime - 2, -1, -1):
                offset[0] = a * V[extraP, i]
                offset[-1] = c * V[-1 - extraP, i]
                V_jump = V[extraP + 1 : -extraP - 1, i + 1] + dt * signal.convolve(
                    V[:, i + 1], nu[::-1], mode="valid", method="fft"
                )
                V[extraP + 1 : -extraP - 1, i] = np.maximum(DD.solve(V_jump - offset), Payoff[extraP + 1 : -extraP - 1])

        X0 = np.log(self.S0)  # current log-price
        self.S_vec = np.exp(x[extraP + 1 : -extraP - 1])  # vector of S
        self.price = np.interp(X0, x, V[:, 0])
        self.price_vec = V[extraP + 1 : -extraP - 1, 0]
        self.mesh = V[extraP + 1 : -extraP - 1, :]

        if Time is True:
            elapsed = time() - t_init
            return self.price, elapsed
        else:
            return self.price

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
                    price_surface[j, i] = self.FFT(self.K)
            
            # Plotting the surface
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(K_grid, T_grid, price_surface, cmap='viridis')
            
            ax.set_xlabel('Strike Price (K)')
            ax.set_ylabel('Time to Maturity (T)')
            ax.set_zlabel('Option Price')
            ax.set_title(f'Merton Jump Diffusion {self.payoff.capitalize()} Option Price Surface')
            
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
            price_surface = gaussian_filter(price_surface, sigma=0.75)
        valid_vals = price_surface[~np.isnan(price_surface)]
        
        if valid_vals.size > 0:
            zmin = np.percentile(valid_vals, 0.01)  
            zmax = np.percentile(valid_vals, 99.99) 
         
        # Create the 3D plot.
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(K_grid, T_grid, price_surface, cmap='viridis')
        
        ax.set_xlabel('Strike Price (K)')
        ax.set_ylabel('Time to Maturity (T)')
        ax.set_zlabel('Implied Volatility')
        ax.set_title('Merton Jump Diffusion Implied Volatility Surface')
        ax.set_zlim(zmin, zmax)
       
    
        # plt.show()
        return fig

    def priceColorGrid(self, S_range, sigma_range, num_S=15, num_sigma=15, show_pl=False):
        """
        Generate a discrete color grid of European option prices or P/L using the Black-Scholes model, 
        with values displayed on each cell.

        Args:
            S_range: List or array of spot prices (S).
            sigma_range: List or array of implied volatilities (sigma).
            num_S: Number of discrete spot prices.
            num_sigma: Number of discrete volatilities.
            show_pl: Boolean flag to toggle between showing option prices or P/L.

        Returns:
            fig: Matplotlib figure object.
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
                sigma = sigma_grid[i, j]
                if show_pl:
                    # Compute P/L as the difference between price and intrinsic value
                    intrinsic_value = max(0, (S - self.K) if self.payoff == 'call' else (self.K - S))
                    values[i, j] = self.MertonClosedForm(self.payoff, S, self.K, self.T, self.r, sigma, self.muJ, self.sigJ, self.lam) - intrinsic_value
                else:
                    # Get the price
                    values[i, j] = self.MertonClosedForm(self.payoff, S, self.K, self.T, self.r, sigma, self.muJ, self.sigJ, self.lam)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        cmap = plt.get_cmap('RdYlGn' if show_pl else 'viridis')  
        c = ax.pcolormesh(S_grid, sigma_grid, values, cmap=cmap, shading='auto', edgecolors='k', linewidth=0.5)
        fig.colorbar(c, label='Option Price' if not show_pl else 'P/L')
        
        # Annotate each cell with the value
        for i in range(S_grid.shape[0]):
            for j in range(S_grid.shape[1]):
                ax.text(S_grid[i, j], sigma_grid[i, j], f'{values[i, j]:.2f}', ha='center', va='center', fontsize=8, color='black')
        
        # Add labels and title
        ax.set_xlabel('Spot Price (S)')
        ax.set_ylabel('Implied Volatility (sigma)')
        ax.set_title(f"Option {'P/L' if show_pl else 'Price'} Color Grid ({self.payoff.capitalize()} Option)")
        
        return fig  # Return the figure instead of showing it
    
    def calculate_greeks(self, n_max=50, print_=False):
        """
        Calculates the Greeks for the Merton Jump Diffusion model.
        """
        # Ensure n_max is an integer
        n_max = int(n_max)

        # Adjusted risk-free rate for jump risk
        r_adj = self.r - self.lam * (np.exp(self.muJ + 0.5 * self.sigJ**2) - 1)

        delta, gamma, theta, vega, rho = 0, 0, 0, 0, 0

        # Summation over the number of jumps
        for n in range(n_max + 1):
            # Poisson probability of n jumps
            poisson_prob = np.exp(-self.lam * self.T) * (self.lam * self.T)**n / factorial(n)

            # Adjusted volatility and drift
            sigma_n = np.sqrt(self.sig**2 + n * self.sigJ**2 / self.T)
            r_n = r_adj + n * (self.muJ / self.T)

            # Black-Scholes parameters
            d1 = (np.log(self.S0 / self.K) + (r_n + 0.5 * sigma_n**2) * self.T) / (sigma_n * np.sqrt(self.T))
            d2 = d1 - sigma_n * np.sqrt(self.T)

            # Greeks calculations
            nd1 = norm.pdf(d1)
            if self.payoff.lower() == "call":
                delta_bs = norm.cdf(d1)
                theta_bs = (-self.S0 * nd1 * sigma_n / (2 * np.sqrt(self.T)) - r_n * self.K * np.exp(-r_n * self.T) * norm.cdf(d2))
                rho_bs = self.K * self.T * np.exp(-r_n * self.T) * norm.cdf(d2)
            elif self.payoff.lower() == "put":
                delta_bs = -norm.cdf(-d1)
                theta_bs = (-self.S0 * nd1 * sigma_n / (2 * np.sqrt(self.T)) + r_n * self.K * np.exp(-r_n * self.T) * norm.cdf(-d2))
                rho_bs = -self.K * self.T * np.exp(-r_n * self.T) * norm.cdf(-d2)
            else:
                raise ValueError("Invalid option type. Use 'call' or 'put'.")

            gamma_bs = nd1 / (self.S0 * sigma_n * np.sqrt(self.T))
            vega_bs = self.S0 * nd1 * np.sqrt(self.T)

            # Aggregate Greeks
            delta += poisson_prob * delta_bs
            gamma += poisson_prob * gamma_bs
            theta += poisson_prob * theta_bs
            vega += poisson_prob * vega_bs
            rho += poisson_prob * rho_bs

        data = {
            "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
            "Value": [delta, gamma, vega, theta, rho],
        }
        df = pd.DataFrame(data).set_index("Greek")
        return df

    def plot_stockpaths(self, M=1000, N=10):
        size = (M, N)
        dt = self.T / M 
        
        poi_rv = np.multiply(
            np.random.poisson(self.lam * dt, size=size),
            np.random.normal(self.muJ, self.sigJ, size=size)
        ).cumsum(axis=0)
        
        geo = np.cumsum(
            ((self.r - self.sig**2 / 2 - self.lam * (self.muJ + self.sigJ**2 * 0.5)) * dt +
            self.sig * np.sqrt(dt) * np.random.normal(size=size)), axis=0
        )
        
        paths = np.exp(geo + poi_rv) * self.S0

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(paths)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Asset Price')
        ax.set_title(f'Merton Jump Diffusion Price Paths ({N} Paths)')

        return fig

    def plot_density(self,N=100000):
        np.random.seed(seed=42)
        W = ss.norm.rvs(0, 1, N)  # The normal RV vector
        P = ss.poisson.rvs(self.lam * self.T, size=N)  # Poisson random vector
        Jumps = np.asarray([ss.norm.rvs(self.muJ, self.sigJ, i).sum() for i in P])  # Jumps vector
        X_T = self.r * self.T + np.sqrt(self.T) * self.sig * W + Jumps  # Merton process
        cf_M_b = partial(cf_mert, t=self.T, mu=self.r, sig=self.sig, lam=self.lam, muJ=self.muJ, sigJ=self.sigJ)
        x = np.linspace(X_T.min(), X_T.max(), 500)
        y = np.linspace(-3, 5, 50)
        
        fig = plt.figure(figsize=(15, 6))
        ax = fig.add_subplot(111)
        ax.plot(x, Merton_pdf(x, self.T, self.r, self.sig, self.lam, self.muJ, self.sigJ), 
                color="r", label="Merton density")
        ax.plot(y, [Gil_Pelaez_pdf(i, cf_M_b, np.inf) for i in y], "p", label="Fourier inversion")
        ax.hist(X_T, density=True, bins=200, facecolor="LightBlue", label="frequencies of X_T")
        ax.legend()
        ax.set_title("Merton Jump Diffusion Histogram")
        return fig

    def plot_qq(self, N=100000):
        np.random.seed(seed=42)
        W = ss.norm.rvs(0, 1, N)  # The normal RV vector
        P = ss.poisson.rvs(self.lam * self.T, size=N)  # Poisson random vector
        Jumps = np.asarray([ss.norm.rvs(self.muJ, self.sigJ, i).sum() for i in P])  # Jumps vector
        X_T = self.r * self.T + np.sqrt(self.T) * self.sig * W + Jumps  # Merton process

        # Compute statistics
        median_val = np.median(X_T)
        mean_val = np.mean(X_T)
        std_val = np.std(X_T)
        skew_val = ss.skew(X_T)
        kurtosis_val = ss.kurtosis(X_T)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        qqplot(X_T, line="s", ax=ax)
        ax.set_title("Merton Jump Diffusion Q-Q Plot")

        # Add text with statistics to the plot
        stats_text = (f"Median: {median_val:.4f}\n"
                    f"Mean: {mean_val:.4f}\n"
                    f"Std Dev: {std_val:.4f}\n"
                    f"Skewness: {skew_val:.4f}\n"
                    f"Kurtosis: {kurtosis_val:.4f}")

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
        return fig

class Merton_process:
    """
    Class for the Merton process:
    r = risk free constant rate
    sig = constant diffusion coefficient
    lam = jump activity
    muJ = jump mean
    sigJ = jump standard deviation
    """

    def __init__(self, r=0.1, sig=0.2, lam=0.8, muJ=0, sigJ=0.5):
        self.r = r
        self.lam = lam
        self.muJ = muJ
        if sig < 0 or sigJ < 0:
            raise ValueError("sig and sigJ must be positive")
        else:
            self.sig = sig
            self.sigJ = sigJ

        # moments
        self.var = self.sig**2 + self.lam * self.sigJ**2 + self.lam * self.muJ**2
        self.skew = self.lam * (3 * self.sigJ**2 * self.muJ + self.muJ**3) / self.var ** (1.5)
        self.kurt = self.lam * (3 * self.sigJ**3 + 6 * self.sigJ**2 * self.muJ**2 + self.muJ**4) / self.var**2

    def exp_RV(self, S0, T, N):
        m = self.lam * (np.exp(self.muJ + (self.sigJ**2) / 2) - 1)  # coefficient m
        W = ss.norm.rvs(0, 1, N)  # The normal RV vector
        P = ss.poisson.rvs(self.lam * T, size=N)  # Poisson random vector (number of jumps)
        Jumps = np.asarray([ss.norm.rvs(self.muJ, self.sigJ, ind).sum() for ind in P])  # Jumps vector
        S_T = S0 * np.exp(
            (self.r - 0.5 * self.sig**2 - m) * T + np.sqrt(T) * self.sig * W + Jumps
        )  # Martingale exponential Merton
        return S_T.reshape((N, 1))

class Option_param:
        def __init__(self, S0, K, T, exercise="European", payoff="call"):
            self.S0 = S0  # Initial stock price
            self.K = K  # Strike price
            self.T = T  # Time to maturity
            
                        
            if exercise == "European" or exercise == "American":
                self.exercise = exercise
            else:
                raise ValueError("invalid type. Set 'European' or 'American'")

            if payoff == "call" or payoff == "put":
                self.payoff = payoff
            else:
                raise ValueError("invalid type. Set 'call' or 'put'")

class process_info:
    def __init__(self, r, sig, lam, muJ, sigJ, exp_RV):
        self.r = r  # interest rate
        self.sig = sig  # diffusion coefficient
        self.lam = lam  # jump activity
        self.muJ = muJ  # jump mean
        self.sigJ = sigJ    # jump std
        self.exp_RV = exp_RV    # function to generate exponential Merton Random Variables


if __name__ == "__main__":

    S0 = 100      # Initial stock price
    K = 110       # Strike price
    T = 1.0       # Time to maturity (1 year)
    r = 0.05      # Risk-free rate
    sigma = 0.2   # Volatility of variance
    muJ = -0.15  # Jump Mean
    sigJ = 0.5    # std dev of jumps
    lam = 1.2     # jump activity
    N=100000
    
    strikes = [60, 140]
    maturities = [0.1, 3.0]
    ivols = [0.1, 1]
    Spots = [60,140]
    
    # Create option and process information
    option_info = Option_param(S0, K, T,payoff="put")
    MJD_process = process_info(r, sigma, lam,muJ,sigJ, Merton_process(r, sigma,lam,muJ,sigJ).exp_RV)

    pricer = Merton_pricer(option_info,MJD_process)
    # pricer.priceSurface(strikes,maturities)
    # pricer.plot_IV_surface(strikes,maturities)
    

    pricer.plot_density()
    pricer.plot_qq()



