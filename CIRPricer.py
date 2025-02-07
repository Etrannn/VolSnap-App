import numpy as np
import scipy.stats as ss
import time
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


class CIR_pricer:
    """
    Closed Formula
    """
    def __init__(self, Bond_info, Process_info):
        self.sigma = Process_info.sigma  
        self.theta = Process_info.theta  
        self.kappa = Process_info.kappa  
        self.exp_RV = Process_info.exp_RV
        
        self.r0 = Bond_info.r0 
        self.T = Bond_info.T  

        self.price = 0
        self.S_vec = None
        self.price_vec = None
        self.mesh = None

    def closed_formula_manual(self,r0, T, sigma,kappa,theta):
        gamma = np.sqrt(kappa**2 + 2*sigma**2)
        alpha_term = np.exp(gamma*T) - 1
        numerator_a = 2*gamma*np.exp(((gamma+kappa)*T)/2)
        denominator = 2*gamma + (kappa + gamma)*alpha_term
        A = (numerator_a/denominator)**((2*kappa*theta)/sigma**2)
        numerator_b = 2 * alpha_term
        B = numerator_b/denominator
        Price = A * np.exp(-B * r0)
        return Price

    def closed_formula(self):
        gamma = np.sqrt(self.kappa**2 + 2*self.sigma**2)
        alpha_term = np.exp(gamma*T) - 1
        numerator_a = 2*gamma*np.exp(((gamma+self.kappa)*T)/2)
        denominator = 2*gamma + (self.kappa + gamma)*alpha_term
        A = (numerator_a/denominator)**((2*self.kappa*self.theta)/self.sigma**2)
        numerator_b = 2 * alpha_term
        B = numerator_b/denominator
        Price = A * np.exp(-B * self.r0)
        return Price
    
    def MC(self, M, Err=False, Time=False):
        t_init = time.time()
        OUProcess = CIR_process(self.sigma,self.theta,self.kappa)
        X = OUProcess.path(self.r0,self.T,paths=M)
        
        disc_factor = np.exp(-X.mean(axis=0) * self.T)
        P_MC = np.mean(disc_factor)
        st_err = ss.sem(disc_factor)
        
        if Err is True:
            if Time is True:
                elapsed = time.time() - t_init
                return P_MC, st_err, elapsed
            else:
                return P_MC, st_err
        else:
            if Time is True:
                elapsed = time.time() - t_init
                return P_MC, elapsed
            else:
                return P_MC
    

    def showPaths(self, paths=10):
        N = 20000  # Time steps
        T = self.T
        kappa, theta, sigma, r0 = self.kappa, self.theta, self.sigma, self.r0
        std_asy = self.sigma*np.sqrt(self.theta)/np.sqrt(2*self.kappa)
        T_vec, dt = np.linspace(0, T, N, retstep=True)

        X = np.zeros((N, paths))
        X[0, :] = r0

        W = ss.norm.rvs(loc=0, scale=np.sqrt(dt), size=(N - 1, paths))  # Brownian motion increments

        for t in range(N - 1):
            dX = kappa * (theta - X[t, :]) * dt + sigma * np.sqrt(X[t, :]) * W[t, :]
            X[t + 1, :] = np.maximum(X[t, :] + dX, 0)  # Ensuring non-negative rates

        # Plot results
        fig, ax1 = plt.subplots(figsize=(16, 5))
        ax1.plot(T_vec, X[:, :paths], linewidth=0.5)
        ax1.plot(T_vec, (theta + std_asy) * np.ones_like(T_vec), label="1 asymptotic std dev", color="black")
        ax1.plot(T_vec, (theta - std_asy) * np.ones_like(T_vec), color="black")
        ax1.plot(T_vec, theta * np.ones_like(T_vec), label="Long term mean")
        ax1.legend(loc="upper right")
        ax1.set_title(f"{paths} CIR Short Rate Paths")
        ax1.set_xlabel("Time (T)")
        ax1.set_ylabel("Short Rate (r)")
        
        plt.show()

    def priceSurface(self, r_range, T_range, num_r=15, num_T=15):
        """
        Generate a 3D surface plot for bond prices under the Vasicek model.
        
        Parameters:
        - num_r: Number of short rate points.
        - num_T: Number of maturity points.        
        - K_range: Tuple (K_min, K_max) for strike prices.
        - T_range: Tuple (T_min, T_max) for time to maturity.

        """
        r_values = np.linspace(r_range[0], r_range[1], num_r)
        T_values = np.linspace(T_range[0], T_range[1], num_T)
        r_grid, T_grid = np.meshgrid(r_values, T_values)
        
        price_surface = np.zeros_like(r_grid)
        
        for i in range(num_r):
            for j in range(num_T):
                self.r = r_values[i]
                self.T = T_values[j]
                price_surface[j, i] = self.closed_formula_manual(self.r,self.T,self.sigma,self.kappa,self.theta)
        
        # Plotting the surface
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(r_grid, T_grid, price_surface, cmap='viridis')
        
        ax.set_xlabel('Short Rate ($r_0$)')
        ax.set_ylabel('Time to Maturity (T)')
        ax.set_zlabel('Bond Price')
        ax.set_title(f'CIR Model bond Price Surface')
        
        plt.show()
    
    def plotTermStructure(self,max_T=30):
        T_values = np.linspace(0.1, max_T, 100)  # Avoid T=0 to prevent division by zero
        yields = []

        for T_val in T_values:
            pricer.T = T_val  # Update the time to maturity
            # Compute bond price using the closed-form formula
            price = pricer.closed_formula()
            # Compute the continuously compounded yield
            yield_val = -np.log(price) / T_val
            yields.append(yield_val)

        # Plot the yield curve
        plt.figure(figsize=(10, 6))
        plt.plot(T_values, yields, lw=2, label="Yield Curve")
        plt.xlabel("Time to Maturity (T)")
        plt.ylabel("Yield")
        plt.title("Predicted Term Structure from the CIR Model")
        plt.grid(True)
        plt.legend()
        plt.show()



class CIR_process:
    """
    Class for the CIR process:
    theta = long term mean
    sigma = diffusion coefficient
    kappa = mean reversion coefficient
    """

    def __init__(self, sigma=0.2, theta=-0.1, kappa=0.1):
        self.theta = theta
        if sigma < 0 or kappa < 0:
            raise ValueError("sigma,theta,kappa must be positive")
        elif 2 * kappa * theta < sigma**2:
            raise ValueError("Feller condition is not satisfied: $2 \kappa \theta ≥ \sigma^2$")
        else:
            self.sigma = sigma
            self.kappa = kappa

    def path(self, X0=0, T=1, N=10000, paths=1):
        """
        Produces a matrix of OU process:  X[N, paths]
        X0 = starting point
        N = number of time points (there are N-1 time steps)
        T = Time in years
        paths = number of paths
        """

        dt = T / (N - 1)
        X = np.zeros((N, paths))
        X[0, :] = X0
        W = ss.norm.rvs(loc=0, scale=1, size=(N - 1, paths))

        first_term = self.sigma**2 * X0 / self.kappa * (np.exp(-self.kappa*dt) - np.exp(-self.kappa*dt))
        second_term = self.sigma**2 * self.theta / (2 * self.kappa) * (1 - np.exp(-self.kappa * dt))
        std_dt = np.sqrt(first_term + second_term**2)
        for t in range(0, N - 1):
            X[t + 1, :] = self.theta + np.exp(-self.kappa * dt) * (X[t, :] - self.theta) + std_dt * np.sqrt(X[t, :])* W[t, :]

        return X

    
class Bond_info:
    def __init__(self, r0, T):
        self.r0 = r0  
        self.T = T  

class Process_info:
    def __init__(self, sigma, theta, kappa, exp_RV):  
        self.sigma = sigma
        self.theta = theta
        self.kappa = kappa
        self.exp_RV = exp_RV


if __name__ == "__main__":

    r0 = 0.01
    T = 5.0
    sigma = 0.1
    theta = 0.04
    kappa = 2.0

    bond_info = Bond_info(r0,T)
    process_info = Process_info(sigma,theta,kappa,CIR_process(sigma,theta,kappa))

    rates = [0.01, 3]
    maturities = [0.01, 5]

    pricer = CIR_pricer(bond_info,process_info)
    pricer.priceSurface(rates,maturities)