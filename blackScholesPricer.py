import numpy as np
import pandas as pd
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
from time import time
import scipy.stats as ss
from functools import partial
from scipy.linalg import norm, solve_triangular
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg import LinAlgError
from scipy.stats import norm
from prereq import fft_Lewis, IV_from_Lewis, Q1, Q2
from blackScholesMCPricer import BlackScholesMCPricer



def Thomas(A, b):
    """
    Solver for the linear equation Ax=b using the Thomas algorithm.
    It is a wrapper of the LAPACK function dgtsv.
    """

    D = A.diagonal(0)
    L = A.diagonal(-1)
    U = A.diagonal(1)

    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expected square matrix")
    if A.shape[0] != b.shape[0]:
        raise ValueError("incompatible dimensions")

    (dgtsv,) = get_lapack_funcs(("gtsv",))
    du2, d, du, x, info = dgtsv(L, D, U, b)

    if info == 0:
        return x
    if info > 0:
        raise LinAlgError("singular matrix: resolution failed at diagonal %d" % (info - 1))

def SOR(A, b, w=1, eps=1e-10, N_max=100):
    """
    Solver for the linear equation Ax=b using the SOR algorithm.
          A = L + D + U
    Arguments:
        L = Strict Lower triangular matrix
        D = Diagonal
        U = Strict Upper triangular matrix
        w = Relaxation coefficient
        eps = tollerance
        N_max = Max number of iterations
    """

    x0 = b.copy()  # initial guess

    if sparse.issparse(A):
        D = sparse.diags(A.diagonal())  # diagonal
        U = sparse.triu(A, k=1)  # Strict U
        L = sparse.tril(A, k=-1)  # Strict L
        DD = (w * L + D).toarray()
    else:
        D = np.eye(A.shape[0]) * np.diag(A)  # diagonal
        U = np.triu(A, k=1)  # Strict U
        L = np.tril(A, k=-1)  # Strict L
        DD = w * L + D

    for i in range(1, N_max + 1):
        x_new = solve_triangular(DD, (w * b - w * U @ x0 - (w - 1) * D @ x0), lower=True)
        if norm(x_new - x0) < eps:
            return x_new
        x0 = x_new
        if i == N_max:
            raise ValueError("Fail to converge in {} iterations".format(i))

def cf_normal(u, mu=1, sig=2):
    """
    Characteristic function of a Normal random variable
    """
    return np.exp(1j * u * mu - 0.5 * u**2 * sig**2)

class BS_pricer:
    """
    Closed Formula.
    Monte Carlo.
    Finite-difference Black-Scholes PDE:
     df/dt + r df/dx + 1/2 sigma^2 d^f/dx^2 -rf = 0
    """

    def __init__(self, Option_info, Process_info):
        """
        Option_info: of type Option_param. It contains (S0,K,T)
                i.e. current price, strike, maturity in years
        Process_info: of type Diffusion_process. It contains (r, mu, sig) i.e.
                interest rate, drift coefficient, diffusion coefficient
        """
        self.r = Process_info.r  # interest rate
        self.sig = Process_info.sig  # diffusion coefficient
        self.S0 = Option_info.S0  # current price
        self.K = Option_info.K  # strike
        self.T = Option_info.T  # maturity in years
        self.exp_RV = Process_info.exp_RV  # function to generate solution of GBM
  

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
    def BlackScholes(payoff="call", S0=100.0, K=100.0, T=1.0, r=0.1, sigma=0.2):
        """Black Scholes closed formula:
        payoff: call or put.
        S0: float.    initial stock/index level.
        K: float strike price.
        T: float maturity (in year fractions).
        r: float constant risk-free short rate.
        sigma: volatility factor in diffusion term."""
        
        
        d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S0 / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))

        if payoff == "call":
            return S0 * ss.norm.cdf(d1) - K * np.exp(-r * T) * ss.norm.cdf(d2)
        elif payoff == "put":
            return K * np.exp(-r * T) * ss.norm.cdf(-d2) - S0 * ss.norm.cdf(-d1)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    @staticmethod
    def vega(sigma, S0, K, T, r):
        """BS vega: derivative of the price with respect to the volatility"""
        d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        return S0 * np.sqrt(T) * ss.norm.pdf(d1)

    def closed_formula(self):
        """
        Black Scholes closed formula:
        """

        d1 = (np.log(self.S0 / self.K) + (self.r + self.sig**2 / 2) * self.T) / (self.sig * np.sqrt(self.T))
        d2 = (np.log(self.S0 / self.K) + (self.r - self.sig**2 / 2) * self.T) / (self.sig * np.sqrt(self.T))

        if self.payoff == "call":
            return self.S0 * ss.norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * ss.norm.cdf(d2)
        elif self.payoff == "put":
            return self.K * np.exp(-self.r * self.T) * ss.norm.cdf(-d2) - self.S0 * ss.norm.cdf(-d1)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")
    
    def adjust_for_dividends(S0,r,T,q):
        """
        - For continuous dividends, adjust using the dividend yield.
        - For discrete dividends, subtract the present value of dividends.
        """
        adjusted_S = S0
        if type(q) == float:
            adjusted_S *= np.exp(-q * T)
            
        elif type(q) == list:
            for t, amount in q:
                if t < T:  # Only consider dividends before option maturity
                    adjusted_S -= amount * np.exp(-r * t)
        else:
            ValueError("Either a Continuous dividend yield (type: float) OR discrete dividends (List of tuples [(time, amount), ...])")
        return adjusted_S
    
    def Fourier_inversion(self):
        """
        Price obtained by inversion of the characteristic function
        """
   

        k = np.log(self.K / self.S0)
        cf_GBM = partial(
            cf_normal,
            mu=(self.r - 0.5 * self.sig**2) * self.T,
            sig=self.sig * np.sqrt(self.T),
        )  # function binding

        if self.payoff == "call":
            call = self.S0 * Q1(k, cf_GBM, np.inf) - self.K * np.exp(-self.r * self.T) * Q2(
                k, cf_GBM, np.inf
            )  # pricing function
            return call
        elif self.payoff == "put":
            put = self.K * np.exp(-self.r * self.T) * (1 - Q2(k, cf_GBM, np.inf)) - self.S0 * (
                1 - Q1(k, cf_GBM, np.inf)
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
        cf_GBM = partial(
            cf_normal,
            mu=(self.r - 0.5 * self.sig**2) * self.T,
            sig=self.sig * np.sqrt(self.T),
        )  # function binding
        if self.payoff == "call":
            return fft_Lewis(K, self.S0, self.r, self.T, cf_GBM, interp="cubic")
        elif self.payoff == "put":  # put-call parity
            return (
                fft_Lewis(K, self.S0, self.r, self.T, cf_GBM, interp="cubic") - self.S0 + K * np.exp(-self.r * self.T)
            )
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def IV_Lewis(self):
        """Implied Volatility from the Lewis formula"""

        cf_GBM = partial(
            cf_normal,
            mu=(self.r - 0.5 * self.sig**2) * self.T,
            sig=self.sig * np.sqrt(self.T),
        )  # function binding
        if self.payoff == "call":
            return IV_from_Lewis(self.K, self.S0, self.T, self.r, cf_GBM)
        elif self.payoff == "put":
            raise NotImplementedError
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def MC(self, N, Err=False, Time=False):
        """
        BS Monte Carlo
        Err = return Standard Error if True
        Time = return execution time if True
        """

        t_init = time()

        S_T = self.exp_RV(self.S0, self.T, N)
        PayOff = self.payoff_f(S_T)
        V = np.mean(np.exp(-self.r * self.T) * PayOff, axis=0)

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

    def PDE_price(self, steps, Time=False, solver="splu"):
        """
        steps = tuple with number of space steps and time steps
        payoff = "call" or "put"
        exercise = "European" or "American"
        Time = Boolean. Execution time.
        Solver = spsolve or splu or Thomas or SOR
        """
        t_init = time()

        Nspace = steps[0]
        Ntime = steps[1]

        S_max = 6 * float(self.K)
        S_min = float(self.K) / 6
        x_max = np.log(S_max)
        x_min = np.log(S_min)
        x0 = np.log(self.S0)  # current log-price

        x, dx = np.linspace(x_min, x_max, Nspace, retstep=True)
        t, dt = np.linspace(0, self.T, Ntime, retstep=True)

        self.S_vec = np.exp(x)  # vector of S
        Payoff = self.payoff_f(self.S_vec)

        V = np.zeros((Nspace, Ntime))
        if self.payoff == "call":
            V[:, -1] = Payoff
            V[-1, :] = np.exp(x_max) - self.K * np.exp(-self.r * t[::-1])
            V[0, :] = 0
        else:
            V[:, -1] = Payoff
            V[-1, :] = 0
            V[0, :] = Payoff[0] * np.exp(-self.r * t[::-1])  # Instead of Payoff[0] I could use K
            # For s to 0, the limiting value is e^(-rT)(K-s)

        sig2 = self.sig**2
        dxx = dx**2
        a = (dt / 2) * ((self.r - 0.5 * sig2) / dx - sig2 / dxx)
        b = 1 + dt * (sig2 / dxx + self.r)
        c = -(dt / 2) * ((self.r - 0.5 * sig2) / dx + sig2 / dxx)

        D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()

        offset = np.zeros(Nspace - 2)

        if solver == "spsolve":
            if self.exercise == "European":
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = spsolve(D, (V[1:-1, i + 1] - offset))
            elif self.exercise == "American":
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = np.maximum(spsolve(D, (V[1:-1, i + 1] - offset)), Payoff[1:-1])
        elif solver == "Thomas":
            if self.exercise == "European":
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = Thomas(D, (V[1:-1, i + 1] - offset))
            elif self.exercise == "American":
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = np.maximum(Thomas(D, (V[1:-1, i + 1] - offset)), Payoff[1:-1])
        elif solver == "SOR":
            if self.exercise == "European":
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = SOR(a, b, c, (V[1:-1, i + 1] - offset), w=1.68, eps=1e-10, N_max=600)
            elif self.exercise == "American":
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = np.maximum(
                        SOR(
                            a,
                            b,
                            c,
                            (V[1:-1, i + 1] - offset),
                            w=1.68,
                            eps=1e-10,
                            N_max=600,
                        ),
                        Payoff[1:-1],
                    )
        elif solver == "splu":
            DD = splu(D)
            if self.exercise == "European":
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = DD.solve(V[1:-1, i + 1] - offset)
            elif self.exercise == "American":
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = np.maximum(DD.solve(V[1:-1, i + 1] - offset), Payoff[1:-1])
        else:
            raise ValueError("Solver is splu, spsolve, SOR or Thomas")

        self.price = np.interp(x0, x, V[:, 0])
        self.price_vec = V[:, 0]
        self.mesh = V

        if Time is True:
            elapsed = time() - t_init
            return self.price, elapsed
        else:
            return self.price

    def priceSurface(self, K_range, T_range, num_K=10, num_T=10):
            """
            Generate a 3D surface plot for option prices under the BSM model.
            
            Parameters:
            - pricer: Instance of BS_pricer.
            - K_range: Tuple (K_min, K_max) for strike prices.
            - T_range: Tuple (T_min, T_max) for time to maturity.
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
                    price_surface[j, i] = self.FFT(K)
            
            # Plotting the surface
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(K_grid, T_grid,  price_surface, cmap='viridis')
            
            ax.set_xlabel('Strike Price (K)')
            ax.set_ylabel('Time to Maturity (T)')
            ax.set_zlabel('Option Price')
            ax.set_title(f'Black Scholes Model {self.payoff.capitalize()} Option Price Surface')
            
            plt.show()

    def plot_IV_surface(pricer, K_range, T_range,num_K=10,num_T=10):
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
                pricer.K = K_values[i]
                pricer.T = T_values[j]
                price_surface[j, i] = pricer.IV_Lewis()
        
        # Plotting the surface
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(K_grid, T_grid, price_surface, cmap='viridis')
        
        ax.set_xlabel('Strike Price (K)')
        ax.set_ylabel('Time to Maturity (T)')
        ax.set_zlabel('Implied Volatility')
        ax.set_title(f'Black Scholes Implied Volatility Surface')
        plt.show()


    def LSM(self, N=10000, paths=10000, order=2):
        """
        Longstaff-Schwartz Method for pricing American options

        N = number of time steps
        paths = number of generated paths
        order = order of the polynomial for the regression
        """
        dt = self.T / (N - 1)  # time interval
        df = np.exp(-self.r * dt)  # discount factor per time interval

        # Generate asset paths using Geometric Brownian Motion
        X0 = np.zeros((paths, 1))
        increments = ss.norm.rvs(
            loc=(self.r - self.sig**2 / 2) * dt,
            scale=np.sqrt(dt) * self.sig,
            size=(paths, N - 1),
        )
        X = np.concatenate((X0, increments), axis=1).cumsum(1)
        S = self.S0 * np.exp(X)

        # Compute the intrinsic values (H) based on the option type.
        if self.payoff == "put":
            H = np.maximum(self.K - S, 0)
        elif self.payoff == "call":
            H = np.maximum(S - self.K, 0)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

        # Initialize the value matrix.
        V = np.zeros_like(H)
        V[:, -1] = H[:, -1]

        # Backward induction using Least Squares
        for t in range(N - 2, 0, -1):
            # Identify paths that are in-the-money at time t.
            good_paths = H[:, t] > 0

            # If no paths are in the money, discount the continuation value.
            if np.sum(good_paths) == 0:
                V[:, t] = V[:, t + 1] * df
                continue

            # Regression: fit a polynomial (of degree 'order') on the in-the-money paths.
            rg = np.polyfit(S[good_paths, t], V[good_paths, t + 1] * df, order)
            # Evaluate the estimated continuation values.
            C = np.polyval(rg, S[good_paths, t])

            # Create an exercise indicator array for all paths.
            exercise = np.zeros(S.shape[0], dtype=bool)
            exercise[good_paths] = H[good_paths, t] > C

            # For exercised paths, set the option value at time t to the intrinsic payoff and zero out future values.
            V[exercise, t] = H[exercise, t]
            V[exercise, t + 1:] = 0
            # For the remaining paths, discount the continuation value.
            discount_path = ~exercise
            V[discount_path, t] = V[discount_path, t + 1] * df

        # Estimate the option price as the discounted average of the values at time 1.
        V0 = np.mean(V[:, 1]) * df
        return V0


    def priceColorGrid(self, S_range, sigma_range,num_S=15,num_sigma=15, show_pl=False):
        """
        Generate a discrete color grid of European option prices or P/L using the Black-Scholes model, with values displayed on each cell.

        Args:
            S_range: List or array of spot prices (S).
            sigma_range: List or array of implied volatilities (sigma).
            r: Risk-free rate.
            K: Strike price.
            T: Time to maturity.
            option_type: Specifies the option type, either 'call' or 'put'.
            show_pl: Boolean flag to toggle between showing option prices or P/L.

        Returns:
            None. Displays a 2D discrete color grid plot.
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
                    intrinsic_value = max(0, (S - K) if self.payoff == 'call' else (K - S))
                    values[i, j] = self.BlackScholes(self.payoff,S,self.K,self.T,self.r,sigma) - intrinsic_value
                else:
                    # Get the price
                    values[i, j] = self.BlackScholes(self.payoff,S,self.K,self.T,self.r,sigma)

        # Plot the 2D discrete color grid
        plt.figure(figsize=(12, 8))
        cmap = plt.get_cmap('RdYlGn' if show_pl else 'viridis')  
        plt.pcolormesh(S_grid, sigma_grid, values, cmap=cmap, shading='auto', edgecolors='k', linewidth=0.5)
        plt.colorbar(label='Option Price' if not show_pl else 'P/L')

        # Annotate each cell with the value
        for i in range(S_grid.shape[0]):
            for j in range(S_grid.shape[1]):
                plt.text(S_grid[i, j], sigma_grid[i, j], f'{values[i, j]:.2f}', ha='center', va='center', fontsize=8, color='black')

        # Add labels and title
        plt.xlabel('Spot Price (S)')
        plt.ylabel('Implied Volatility (sigma)')
        plt.title(f"Option {'P/L' if show_pl else 'Price'} Color Grid ({self.payoff.capitalize()} Option)")
        plt.show()

    def d1(self):
        return (np.log(self.S0 / self.K) + (self.r + self.sig**2 / 2) * self.T) / (self.sig * np.sqrt(self.T))
    
    def d2(self):
        return (np.log(self.S0 / self.K) + (self.r - self.sig**2 / 2) * self.T) / (self.sig * np.sqrt(self.T))    
    
    def calculate_greeks(self):
        d1 = self.d1()
        d2 = self.d2()

        # Delta
        delta = norm.cdf(d1) if self.payoff == "call" else norm.cdf(d1) - 1

        # Gamma
        gamma = norm.pdf(d1) / (self.S0 * self.sig * np.sqrt(self.T))

        # Vega
        vega = self.S0 * norm.pdf(d1) * np.sqrt(self.T) * 0.01

        # Theta
        term1 = -(self.S0 * norm.pdf(d1) * self.sig) / (2 * np.sqrt(self.T))
        if self.payoff == "call":
            term2 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:  # self.payoff == "put"
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
        theta = (term1 + term2) / 365

        # Rho
        if self.payoff == "call":
            rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) * 0.01
        else:  # self.payoff == "put"
            rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) * 0.01

        # Create DataFrame to display Greeks
        data = {
            "Greek": ["Delta: ", "Gamma: ", "Vega: ", "Theta: ", "Rho: "],
            "Value": [delta, gamma, vega, theta, rho],
        }
        df = pd.DataFrame(data).set_index("Greek")
        return df
    
class Diffusion_process:
    """
    Class for the diffusion process:
    r = risk free constant rate
    sig = constant diffusion coefficient
    mu = constant drift
    """

    def __init__(self, r=0.1, sig=0.2, mu=0.1):
        self.r = r
        self.mu = mu
        if sig <= 0:
            raise ValueError("sig must be positive")
        else:
            self.sig = sig

    def exp_RV(self, S0, T, N):
        W = ss.norm.rvs((self.r - 0.5 * self.sig**2) * T, np.sqrt(T) * self.sig, N)
        S_T = S0 * np.exp(W)
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
    def __init__(self, r, sig, exp_RV):
        self.r = r
        self.sig = sig
        self.exp_RV = exp_RV
        

       
  

if __name__ == "__main__":

    S0 = 100      # Initial stock price
    K = 110       # Strike price
    T = 1.0       # Time to maturity (1 year)
    r = 0.05      # Risk-free rate
    sigma = 0.2   # Volatility of variance
    N=100000000
    
    strikes = [60, 140]
    maturities = [0.1, 3.0]   
    ivols = [0.1, 1]
    Spots = [60,140]
    
    # Create option and process information
    option_info = Option_param(S0, K, T,exercise="American",payoff="call")
    BSM_process = process_info(r, sigma, Diffusion_process(r, sigma).exp_RV)

    pricer = BS_pricer(option_info,BSM_process)
    print(pricer.Fourier_inversion())
    print(pricer.closed_formula())
    # pricer.priceSurface(strikes,maturities)
    
    # pricer.priceColorGrid(Spots,ivols,show_pl=False)
    # pricer.priceColorGrid(Spots,ivols,show_pl=True)
    
    # print(pricer.calculate_greeks()) 
    # mcpricer = BlackScholesMCPricer(S0,K,r,T,sigma,N=1000,M=10000)
    # option_value, standard_error = mcpricer.monteCarlo()
    # print(f"Call option value: ${option_value:.4f} ± {standard_error:.4f}")
   

    pde_price, pde_time = pricer.PDE_price(steps=(5000, 4000), Time=True, solver="splu")
    print("European Call PDE Price: {:.4f}".format(pde_price))
    print("PDE computation time: {:.4f} seconds".format(pde_time))
    
    # Test the LSM method. Use smaller values for N and paths for speed in this test.
    lsm_price = pricer.LSM(N=100, paths=1000, order=2)
    print("American LSM Price: {:.4f}".format(lsm_price))
    
    print(pricer.calculate_greeks())