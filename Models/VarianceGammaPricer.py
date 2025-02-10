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
from scipy.optimize import minimize
from prereq import fft_Lewis, IV_from_Lewis, Q1, Q2

def VG_pdf(x, T, c, theta, sigma, kappa):
    """
    Variance Gamma density function
    """
    return (
        2
        * np.exp(theta * (x - c) / sigma**2)
        / (kappa ** (T / kappa) * np.sqrt(2 * np.pi) * sigma * scps.gamma(T / kappa))
        * ((x - c) ** 2 / (2 * sigma**2 / kappa + theta**2)) ** (T / (2 * kappa) - 1 / 4)
        * scps.kv(
            T / kappa - 1 / 2,
            sigma ** (-2) * np.sqrt((x - c) ** 2 * (2 * sigma**2 / kappa + theta**2)),
        )
    )

def cf_VG(u, t=1, mu=0, theta=-0.1, sigma=0.2, kappa=0.1):
    """
    Characteristic function of a Variance Gamma random variable at time t
    mu: additional drift
    theta: Brownian motion drift
    sigma: Brownian motion diffusion
    kappa: Gamma process variance
    """
    return np.exp(t * (1j * mu * u - np.log(1 - 1j * theta * kappa * u + 0.5 * kappa * sigma**2 * u**2) / kappa))

class VG_pricer:
    """
    Closed Formula.
    Monte Carlo.
    Finite-difference PIDE: Explicit-implicit scheme, with Brownian approximation

        0 = dV/dt + (r -(1/2)sig^2 -w) dV/dx + (1/2)sig^2 d^V/dx^2
                 + \int[ V(x+y) nu(dy) ] -(r+lam)V
    """

    def __init__(self, Option_info, Process_info):
        """
        Process_info:  of type VG_process.
        It contains the interest rate r and the VG parameters (sigma, theta, kappa)

        Option_info:  of type Option_param.
        It contains (S0,K,T) i.e. current price, strike, maturity in years
        """
        self.r = Process_info.r  # interest rate
        self.sigma = Process_info.sigma  # VG parameter
        self.theta = Process_info.theta  # VG parameter
        self.kappa = Process_info.kappa  # VG parameter
        self.exp_RV = Process_info.exp_RV  # function to generate exponential VG Random Variables
        self.w = -np.log(1 - self.theta * self.kappa - self.kappa / 2 * self.sigma**2) / self.kappa  # coefficient w

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

    def closed_formula(self):
        """
        VG closed formula.  Put is obtained by put/call parity.
        """

        def Psy(a, b, g):
            f = lambda u: ss.norm.cdf(a / np.sqrt(u) + b * np.sqrt(u)) * u ** (g - 1) * np.exp(-u) / scps.gamma(g)
            result = quad(f, 0, np.inf)
            return result[0]

        # Ugly parameters
        xi = -self.theta / self.sigma**2
        s = self.sigma / np.sqrt(1 + ((self.theta / self.sigma) ** 2) * (self.kappa / 2))
        alpha = xi * s

        c1 = self.kappa / 2 * (alpha + s) ** 2
        c2 = self.kappa / 2 * alpha**2
        d = 1 / s * (np.log(self.S0 / self.K) + self.r * self.T + self.T / self.kappa * np.log((1 - c1) / (1 - c2)))

        # Closed formula
        call = self.S0 * Psy(
            d * np.sqrt((1 - c1) / self.kappa),
            (alpha + s) * np.sqrt(self.kappa / (1 - c1)),
            self.T / self.kappa,
        ) - self.K * np.exp(-self.r * self.T) * Psy(
            d * np.sqrt((1 - c2) / self.kappa),
            (alpha) * np.sqrt(self.kappa / (1 - c2)),
            self.T / self.kappa,
        )

        if self.payoff == "call":
            return call
        elif self.payoff == "put":
            return call - self.S0 + self.K * np.exp(-self.r * self.T)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def Fourier_inversion(self):
        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(self.K / self.S0)  # log moneyness
        cf_VG_b = partial(
            cf_VG,
            t=self.T,
            mu=(self.r - self.w),
            theta=self.theta,
            sigma=self.sigma,
            kappa=self.kappa,
        )

        right_lim = 5000  # using np.inf may create warnings
        if self.payoff == "call":
            call = self.S0 * Q1(k, cf_VG_b, right_lim) - self.K * np.exp(-self.r * self.T) * Q2(
                k, cf_VG_b, right_lim
            )  # pricing function
            return call
        elif self.payoff == "put":
            put = self.K * np.exp(-self.r * self.T) * (1 - Q2(k, cf_VG_b, right_lim)) - self.S0 * (
                1 - Q1(k, cf_VG_b, right_lim)
            )  # pricing function
            return put
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def MC(self, N, Err=False, Time=False):
        """
        Variance Gamma Monte Carlo
        Err = return Standard Error if True
        Time = return execution time if True
        """
        t_init = time()

        S_T = self.exp_RV(self.S0, self.T, N)
        V = scp.mean(np.exp(-self.r * self.T) * self.payoff_f(S_T), axis=0)

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

    def FFT(self, K):
        """
        FFT method. It returns a vector of prices.
        K is an array of strikes
        """
        K = np.array(K)
        cf_VG_b = partial(
            cf_VG,
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
            cf_VG,
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

        S_max = 6 * float(self.K)
        S_min = float(self.K) / 6
        x_max = np.log(S_max)
        x_min = np.log(S_min)

        dev_X = np.sqrt(self.sigma**2 + self.theta**2 * self.kappa)  # std dev VG process

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

        A = self.theta / (self.sigma**2)
        B = np.sqrt(self.theta**2 + 2 * self.sigma**2 / self.kappa) / self.sigma**2

        def levy_m(y):
            """Levy measure VG"""
            return np.exp(A * y - B * np.abs(y)) / (self.kappa * np.abs(y))

        eps = 1.5 * dx  # the cutoff near 0
        lam = (
            quad(levy_m, -(extraP + 1.5) * dx, -eps)[0] + quad(levy_m, eps, (extraP + 1.5) * dx)[0]
        )  # approximated intensity

        def int_w(y):
            """integrator"""
            return (np.exp(y) - 1) * levy_m(y)

        int_s = lambda y: np.abs(y) * np.exp(A * y - B * np.abs(y)) / self.kappa  # avoid division by zero

        w = (
            quad(int_w, -(extraP + 1.5) * dx, -eps)[0] + quad(int_w, eps, (extraP + 1.5) * dx)[0]
        )  # is the approx of omega

        sig2 = quad(int_s, -eps, eps)[0]  # the small jumps variance

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
            nu[i] = quad(levy_m, x_nu[i], x_nu[i + 1])[0]

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
        Generate a 3D surface plot for option prices under the Variance Gamma model.
        
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
        ax.set_title(f'Variance Gamma Model {self.payoff.capitalize()} Option Price Surface')
        
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
        ax.set_title(f'Variance Gamma Implied Volatility Surface')
        plt.show()
    
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
class VG_process:
    """
    Class for the Variance Gamma process:
    r = risk free constant rate
    Using the representation of Brownian subordination, the parameters are:
        theta = drift of the Brownian motion
        sigma = standard deviation of the Brownian motion
        kappa = variance of the of the Gamma process
    """

    def __init__(self, r=0.1, sigma=0.2, theta=-0.1, kappa=0.1):
        self.r = r
        self.c = self.r
        self.theta = theta
        self.kappa = kappa
        if sigma < 0:
            raise ValueError("sigma must be positive")
        else:
            self.sigma = sigma

        # moments
        self.mean = self.c + self.theta
        self.var = self.sigma**2 + self.theta**2 * self.kappa
        self.skew = (2 * self.theta**3 * self.kappa**2 + 3 * self.sigma**2 * self.theta * self.kappa) / (
            self.var ** (1.5)
        )
        self.kurt = (
            3 * self.sigma**4 * self.kappa
            + 12 * self.sigma**2 * self.theta**2 * self.kappa**2
            + 6 * self.theta**4 * self.kappa**3
        ) / (self.var**2)

    def exp_RV(self, S0, T, N):
        w = -np.log(1 - self.theta * self.kappa - self.kappa / 2 * self.sigma**2) / self.kappa  # coefficient w
        rho = 1 / self.kappa
        G = ss.gamma(rho * T).rvs(N) / rho  # The gamma RV
        Norm = ss.norm.rvs(0, 1, N)  # The normal RV
        VG = self.theta * G + self.sigma * np.sqrt(G) * Norm  # VG process at final time G
        S_T = S0 * np.exp((self.r - w) * T + VG)  # Martingale exponential VG
        return S_T.reshape((N, 1))

    def path(self, T=1, N=10000, paths=1):
        """
        Creates Variance Gamma paths
        N = number of time points (time steps are N-1)
        paths = number of generated paths
        """
        dt = T / (N - 1)  # time interval
        X0 = np.zeros((paths, 1))
        G = ss.gamma(dt / self.kappa, scale=self.kappa).rvs(size=(paths, N - 1))  # The gamma RV
        Norm = ss.norm.rvs(loc=0, scale=1, size=(paths, N - 1))  # The normal RV
        increments = self.c * dt + self.theta * G + self.sigma * np.sqrt(G) * Norm
        X = np.concatenate((X0, increments), axis=1).cumsum(1)
        return X

    def fit_from_data(self, data, dt=1, method="Nelder-Mead"):
        """
        Fit the 4 parameters of the VG process using MM (method of moments),
        Nelder-Mead, L-BFGS-B.

        data (array): datapoints
        dt (float):     is the increment time

        Returns (c, theta, sigma, kappa)
        """
        X = data
        sigma_mm = np.std(X) / np.sqrt(dt)
        kappa_mm = dt * ss.kurtosis(X) / 3
        theta_mm = np.sqrt(dt) * ss.skew(X) * sigma_mm / (3 * kappa_mm)
        c_mm = np.mean(X) / dt - theta_mm

        def log_likely(x, data, T):
            return (-1) * np.sum(np.log(VG_pdf(data, T, x[0], x[1], x[2], x[3])))

        if method == "L-BFGS-B":
            if theta_mm < 0:
                result = minimize(
                    log_likely,
                    x0=[c_mm, theta_mm, sigma_mm, kappa_mm],
                    method="L-BFGS-B",
                    args=(X, dt),
                    tol=1e-8,
                    bounds=[[-0.5, 0.5], [-0.6, -1e-15], [1e-15, 1], [1e-15, 2]],
                )
            else:
                result = minimize(
                    log_likely,
                    x0=[c_mm, theta_mm, sigma_mm, kappa_mm],
                    method="L-BFGS-B",
                    args=(X, dt),
                    tol=1e-8,
                    bounds=[[-0.5, 0.5], [1e-15, 0.6], [1e-15, 1], [1e-15, 2]],
                )
            print(result.message)
        elif method == "Nelder-Mead":
            result = minimize(
                log_likely,
                x0=[c_mm, theta_mm, sigma_mm, kappa_mm],
                method="Nelder-Mead",
                args=(X, dt),
                options={"disp": False, "maxfev": 3000},
                tol=1e-8,
            )
            print(result.message)
        elif "MM":
            self.c, self.theta, self.sigma, self.kappa = (
                c_mm,
                theta_mm,
                sigma_mm,
                kappa_mm,
            )
            return
        self.c, self.theta, self.sigma, self.kappa = result.x

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
    VGprocess = process_info(r, sigma, theta, kappa, VG_process(r, sigma,theta, kappa).exp_RV)

    pricer = VG_pricer(option_info,VGprocess)
    print(pricer.calculate_greeks())
    # print(pricer.calculate_greeks())
    # print(pricer.Fourier_inversion())
    # pricer.priceSurface(strikes,maturities)
    # pricer.plot_IV_surface(strikes,maturities)
    

