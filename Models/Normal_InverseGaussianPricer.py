from scipy import sparse
from scipy.sparse.linalg import splu
from time import time
import numpy as np
import pandas as pd
import scipy as scp
from scipy import signal
from scipy.integrate import quad
import scipy.stats as ss
import scipy.special as scps
import matplotlib.pyplot as plt
from functools import partial
from scipy.ndimage import gaussian_filter
from statsmodels.graphics.gofplots import qqplot
from prereq import fft_Lewis, IV_from_Lewis, Q1, Q2, Gil_Pelaez_pdf,NIG_pdf

def cf_NIG(u, t=1, mu=0, theta=-0.1, sigma=0.2, kappa=0.1):
    """
    Characteristic function of a Normal Inverse Gaussian random variable at time t
    mu: additional drift
    theta: Brownian motion drift
    sigma: Brownian motion diffusion
    kappa: Inverse Gaussian process variance
    """
    return np.exp(
        t * (1j * mu * u + 1 / kappa - np.sqrt(1 - 2j * theta * kappa * u + kappa * sigma**2 * u**2) / kappa)
    )

class NIG_pricer:
    """
    Closed Formula.
    Monte Carlo.
    Finite-difference PIDE: Explicit-implicit scheme, with Brownian approximation

        0 = dV/dt + (r -(1/2)sig^2 -w) dV/dx + (1/2)sig^2 d^V/dx^2
                 + \int[ V(x+y) nu(dy) ] -(r+lam)V
    """

    def __init__(self, Option_info, Process_info):
        """
        Process_info:  of type NIG_process. It contains the interest rate r
        and the NIG parameters (sigma, theta, kappa)

        Option_info:  of type Option_param.
        It contains (S0,K,T) i.e. current price, strike, maturity in years
        """
        self.r = Process_info.r  # interest rate
        self.sigma = Process_info.sigma  # NIG parameter
        self.theta = Process_info.theta  # NIG parameter
        self.kappa = Process_info.kappa  # NIG parameter
        self.exp_RV = Process_info.exp_RV  # function to generate exponential NIG Random Variables
        self.w = (1 - np.sqrt(1 - 2 * self.theta * self.kappa - self.kappa * self.sigma**2)) / self.kappa  # martingale correction
        
        self.S0 = Option_info.S0  # current price
        self.K = Option_info.K  # strike
        self.T = Option_info.T  # maturity in years

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

    def Fourier_inversion(self):
        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(self.K / self.S0)  # log moneyness
        
        cf_NIG_b = partial(
            cf_NIG,
            t=self.T,
            mu=(self.r - self.w),
            theta=self.theta,
            sigma=self.sigma,
            kappa=self.kappa,
        )

        if self.payoff == "call":
            call = self.S0 * Q1(k, cf_NIG_b, np.inf) - self.K * np.exp(-self.r * self.T) * Q2(
                k, cf_NIG_b, np.inf
            )  # pricing function
            return call
        elif self.payoff == "put":
            put = self.K * np.exp(-self.r * self.T) * (1 - Q2(k, cf_NIG_b, np.inf)) - self.S0 * (
                1 - Q1(k, cf_NIG_b, np.inf)
            )  # pricing function
            return put
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def MC(self, N, Err=False, Time=False):
        """
        NIG Monte Carlo
        Err = return Standard Error if True
        Time = return execution time if True
        """
        t_init = time()

        S_T = self.exp_RV(self.S0, self.T, N)
        V = np.mean(np.exp(-self.r * self.T) * self.payoff_f(S_T))

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

    def NIG_measure(self, x):
        A = self.theta / (self.sigma**2)
        B = np.sqrt(self.theta**2 + self.sigma**2 / self.kappa) / self.sigma**2
        C = np.sqrt(self.theta**2 + self.sigma**2 / self.kappa) / (np.pi * self.sigma * np.sqrt(self.kappa))
        return C / np.abs(x) * np.exp(A * (x)) * scps.kv(1, B * np.abs(x))

    def FFT(self, K):
        """
        FFT method. It returns a vector of prices.
        K is an array of strikes
        """
        K = np.array(K)
        cf_VG_b = partial(
            cf_NIG,
            t=self.T,
            mu=(self.r - self.w),
            theta=self.theta,
            sigma=self.sigma,
            kappa=self.kappa,
        )

        if self.payoff == "call":
            return fft_Lewis(K, self.S0, self.r, self.T, cf_VG_b, interp="cubic")
        elif self.payoff == "put":  # put-call parity
            return (
                fft_Lewis(K, self.S0, self.r, self.T, cf_VG_b, interp="cubic") - self.S0 + K * np.exp(-self.r * self.T)
            )
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def IV_Lewis(self):
        """Implied Volatility from the Lewis formula"""

        cf_VG_b = partial(
            cf_NIG,
            t=self.T,
            mu=(self.r - self.w),
            theta=self.theta,
            sigma=self.sigma,
            kappa=self.kappa,
        )

        if self.payoff == "call":
            return IV_from_Lewis(self.K, self.S0, self.T, self.r, cf_VG_b)
        elif self.payoff == "put":
            raise NotImplementedError
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")
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

        S_max = 2000 * float(self.K)
        S_min = float(self.K) / 2000
        x_max = np.log(S_max)
        x_min = np.log(S_min)

        dev_X = np.sqrt(self.sigma**2 + self.theta**2 * self.kappa)  # std dev NIG process

        dx = (x_max - x_min) / (Nspace - 1)
        extraP = int(np.floor(7 * dev_X / dx))  # extra points beyond the B.C.
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

        eps = 1.5 * dx  # the cutoff near 0
        lam = (
            quad(self.NIG_measure, -(extraP + 1.5) * dx, -eps)[0] + quad(self.NIG_measure, eps, (extraP + 1.5) * dx)[0]
        )  # approximated intensity

        def int_w(y):
            return (np.exp(y) - 1) * self.NIG_measure(y)

        def int_s(y):
            return y**2 * self.NIG_measure(y)

        w = quad(int_w, -(extraP + 1.5) * dx, -eps)[0] + quad(int_w, eps, (extraP + 1.5) * dx)[0]  # is the approx of w
        sig2 = quad(int_s, -eps, eps, points=0)[0]  # the small jumps variance

        dxx = dx * dx
        a = (dt / 2) * ((self.r - w - 0.5 * sig2) / dx - sig2 / dxx)
        b = 1 + dt * (sig2 / dxx + self.r + lam)
        c = -(dt / 2) * ((self.r - w - 0.5 * sig2) / dx + sig2 / dxx)
        D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()
        DD = splu(D)

        nu = np.zeros(2 * extraP + 3)  # Lévy measure vector
        x_med = extraP + 1  # middle point in nu vector
        x_nu = np.linspace(-(extraP + 1 + 0.5) * dx, (extraP + 1 + 0.5) * dx, 2 * (extraP + 2))  # integration domain
        for i in range(len(nu)):
            if (i == x_med) or (i == x_med - 1) or (i == x_med + 1):
                continue
            nu[i] = quad(self.NIG_measure, x_nu[i], x_nu[i + 1])[0]

        if self.exercise == "European":
            # Backward iteration
            for i in range(Ntime - 2, -1, -1):
                offset[0] = a * V[extraP, i]
                offset[-1] = c * V[-1 - extraP, i]
                V_jump = V[extraP + 1 : -extraP - 1, i + 1] + dt * signal.convolve(
                    V[:, i + 1], nu[::-1], mode="valid", method="auto"
                )
                V[extraP + 1 : -extraP - 1, i] = DD.solve(V_jump - offset)
        elif self.exercise == "American":
            for i in range(Ntime - 2, -1, -1):
                offset[0] = a * V[extraP, i]
                offset[-1] = c * V[-1 - extraP, i]
                V_jump = V[extraP + 1 : -extraP - 1, i + 1] + dt * signal.convolve(
                    V[:, i + 1], nu[::-1], mode="valid", method="auto"
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
                price_surface[j, i] = self.FFT(self.K)
        
        # Plotting the surface
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(K_grid, T_grid, price_surface, cmap='viridis')
        
        ax.set_xlabel('Strike Price (K)')
        ax.set_ylabel('Time to Maturity (T)')
        ax.set_zlabel('Option Price')
        ax.set_title(f'Normal Inverse Gaussian Model {self.payoff.capitalize()} Option Price Surface')
        
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
        ax.set_title('Normal Inverse Gaussian Implied Volatility Surface')
        ax.set_zlim(zmin, zmax)
       
    
        # plt.show()
        return fig

    def plot_density(self,N=100000):
        lam = self.T**2 / self.kappa  # scale
        mus = self.T / lam  # scaled mu
        IG = ss.invgauss.rvs(mu=mus, scale=lam, size=N)  # The IG RV
        Norm = ss.norm.rvs(0, 1, N)  # The normal RV
        X = self.theta * IG + self.sigma * np.sqrt(IG) * Norm
        cf_NIG_b = partial(cf_NIG, t=self.T, mu=0, theta=self.theta, sigma=self.sigma, kappa=self.kappa)
        x = np.linspace(X.min(), X.max(), 500)
        y = np.linspace(-2, 1, 30)

        fig = plt.figure(figsize=(16, 5))
        ax = fig.add_subplot(111)
        ax.plot(x, NIG_pdf(x, self.T, 0, self.theta, self.sigma, self.kappa), color="r", label="NIG density")
        ax.plot(y, [Gil_Pelaez_pdf(i, cf_NIG_b, np.inf) for i in y], "p", label="Fourier inversion")
        ax.hist(X, density=True, bins=200, facecolor="LightBlue", label="frequencies of X")
        ax.legend()
        ax.set_title("Normal Inverse Gaussian Histogram")
        return fig
            
    def plot_qq(self,N=100000):
        lam = self.T**2 / self.kappa  # scale
        mus = self.T / lam  # scaled mu
        IG = ss.invgauss.rvs(mu=mus, scale=lam, size=N)  # The IG RV
        Norm = ss.norm.rvs(0, 1, N)  # The normal RV
        X = self.theta * IG + self.sigma * np.sqrt(IG) * Norm
        # Compute statistics
        median_val = np.median(X)
        mean_val = np.mean(X)
        std_val = np.std(X)
        skew_val = ss.skew(X)
        kurtosis_val = ss.kurtosis(X)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        qqplot(X, line="s", ax=ax)
        ax.set_title("Normal Inverse Gaussian  Q-Q Plot")

        # Add text with statistics to the plot
        stats_text = (f"Median: {median_val:.4f}\n"
                    f"Mean: {mean_val:.4f}\n"
                    f"Std Dev: {std_val:.4f}\n"
                    f"Skewness: {skew_val:.4f}\n"
                    f"Kurtosis: {kurtosis_val:.4f}")

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
        return fig

    def calculate_greeks(self, h=1e-4):
        orig_S0   = self.S0
        orig_r    = self.r
        orig_sigma = self.sigma
        orig_T    = self.T

        base_price = self.Fourier_inversion()

        self.S0 = orig_S0 + h
        price_up = self.Fourier_inversion()
        self.S0 = orig_S0 - h
        price_down = self.Fourier_inversion()
        
        delta = (price_up - price_down) / (2 * h)
        gamma = (price_up - 2 * base_price + price_down) / (h**2)
        
        self.S0 = orig_S0

        self.sigma = orig_sigma + h
        price_up = self.Fourier_inversion()
        self.sigma = orig_sigma - h
        price_down = self.Fourier_inversion()
        vega = (price_up - price_down) / (2 * h)
        self.sigma = orig_sigma

        self.r = orig_r + h
        price_up = self.Fourier_inversion()
        self.r = orig_r 
        price_down = self.Fourier_inversion()
        rho = (price_up - price_down) / (h)
        self.r = orig_r

        self.T = orig_T - h
        price_up = self.Fourier_inversion()
        self.T = orig_T + h
        price_down = self.Fourier_inversion()
        theta = (price_up - price_down) / (2 * h)
        self.T = orig_T

        data = {
            "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
            "Value": [delta, gamma, vega, theta, rho],
        }
        df = pd.DataFrame(data).set_index("Greek")
        return df

class NIG_process:
    """
    Class for the Normal Inverse Gaussian process:
    r = risk free constant rate
    Using the representation of Brownian subordination, the parameters are:
        theta = drift of the Brownian motion
        sigma = standard deviation of the Brownian motion
        kappa = variance of the of the Gamma process
    """

    def __init__(self, r=0.1, sigma=0.2, theta=-0.1, kappa=0.1):
        self.r = r
        self.theta = theta
        if sigma < 0 or kappa < 0:
            raise ValueError("sigma and kappa must be positive")
        else:
            self.sigma = sigma
            self.kappa = kappa

        # moments
        self.var = self.sigma**2 + self.theta**2 * self.kappa
        self.skew = (3 * self.theta**3 * self.kappa**2 + 3 * self.sigma**2 * self.theta * self.kappa) / (
            self.var ** (1.5)
        )
        self.kurt = (
            3 * self.sigma**4 * self.kappa
            + 18 * self.sigma**2 * self.theta**2 * self.kappa**2
            + 15 * self.theta**4 * self.kappa**3
        ) / (self.var**2)

    def exp_RV(self, S0, T, N):
        lam = T**2 / self.kappa  # scale for the IG process
        mu_s = T / lam  # scaled mean
        w = (1 - np.sqrt(1 - 2 * self.theta * self.kappa - self.kappa * self.sigma**2)) / self.kappa
        IG = ss.invgauss.rvs(mu=mu_s, scale=lam, size=N)  # The IG RV
        Norm = ss.norm.rvs(0, 1, N)  # The normal RV
        X = self.theta * IG + self.sigma * np.sqrt(IG) * Norm  # NIG random vector
        S_T = S0 * np.exp((self.r - w) * T + X)  # exponential dynamics
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
    def __init__(self, r, sigma, theta, kappa, exp_RV):
        self.r = r  # interest rate
        self.sigma = sigma  # diffusion coefficient
        self.theta = theta
        self.kappa = kappa
        self.exp_RV = exp_RV



if __name__ == "__main__":

    S0 = 100      # Initial stock price
    K = 110       # Strike price
    T = 1.0       # Time to maturity (1 year)
    r = 0.05      # Risk-free rate
    sigma = 0.2   # Volatility of variance
    theta = 0.8   # Drift of the Brownian motion
    kappa = 0.5   # variance of the of the Gamma process
    N=100000
    
    strikes = [60, 140]
    maturities = [0.1, 3.0]   
    ivols = [0.1, 1]
    Spots = [60,140]
    
    # Create option and process information
    option_info = Option_param(S0, K, T,payoff="call")
    NIGprocess = process_info(r, sigma, theta, kappa, NIG_process(r, sigma,theta, kappa).exp_RV)

    pricer = NIG_pricer(option_info,NIGprocess)
    # print(pricer.calculate_greeks())
    # pricer.priceSurface(strikes,maturities)
    # pricer.plot_IV_surface(strikes,maturities)

    print(pricer.MC(N,True,True))

