import numpy as np
import scipy.stats as ss
import time
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

class Vasicek_pricer:
    """
    Closed Formula.
    Monte Carlo.
    Finite-difference PIDE: Explicit-implicit scheme
    """

    def __init__(self, Bond_info, Process_info):
        """
        Process_info:  of type OU_process. It contains the interest rate r
        and the OU parameters (sigma, theta, kappa)

        Bond_info:  of type Option_param.
        It contains (r0,T) i.e. Initial short rate and maturity in years
        """
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
        B = 1 / kappa * (1 - np.exp(-kappa * T))
        A = np.exp((theta - sigma**2 / (2 * kappa**2)) * (B - T) - sigma**2 / (4 * kappa) * B**2)
        Price = A * np.exp(-B * r0)
        return Price
    

    def closed_formula(self):
        B = 1 / self.kappa * (1 - np.exp(-self.kappa * self.T))
        A = np.exp((self.theta - self.sigma**2 / (2 * self.kappa**2)) * (B - self.T) - self.sigma**2 / (4 * self.kappa) * B**2)
        Price = A * np.exp(-B * self.r0)
        return Price
    
    def MC(self, M, Err=False, Time=False):
        t_init = time.time()
        OUProcess = OU_process(self.sigma,self.theta,self.kappa)
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
            
    def PDE_price(self,Nspace = 5000,Ntime = 5000,r_max = 3,r_min = -0.8):
        
        r, dr = np.linspace(r_min, r_max, Nspace, retstep=True)  # space discretization
        T_array, Dt = np.linspace(0, self.T, Ntime, retstep=True)  # time discretization
        Payoff = 1  # Bond payoff

        V = np.zeros((Nspace, Ntime))  # grid initialization
        offset = np.zeros(Nspace - 2)  # vector to be used for the boundary terms
        V[:, -1] = Payoff  # terminal conditions
        V[-1, :] = np.exp(-r[-1] * (self.T - T_array))  # lateral boundary condition
        V[0, :] = np.exp(-r[0] * (self.T - T_array))  # lateral boundary condition

        # construction of the tri-diagonal matrix D
        sig2 = self.sigma * self.sigma
        drr = dr * dr
        max_part = np.maximum(self.kappa * (self.theta - r[1:-1]), 0)  # upwind positive part
        min_part = np.minimum(self.kappa * (self.theta - r[1:-1]), 0)  # upwind negative part

        a = min_part * (Dt / dr) - 0.5 * (Dt / drr) * sig2
        b = 1 + Dt * r[1:-1] + (Dt / drr) * sig2 + Dt / dr * (max_part - min_part)
        c = -max_part * (Dt / dr) - 0.5 * (Dt / drr) * sig2

        a0 = a[0]
        cM = c[-1]  # boundary terms
        aa = a[1:]
        cc = c[:-1]  # upper and lower diagonals
        D = sparse.diags([aa, b, cc], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()  # matrix D

        for n in range(Ntime - 2, -1, -1):
            # backward computation
            offset[0] = a0 * V[0, n]
            offset[-1] = cM * V[-1, n]
            V[1:-1, n] = spsolve(D, (V[1:-1, n + 1] - offset))

        # finds the bond price with initial value r0
        Price = np.interp(self.r0, r, V[:, 0])
        return Price
    
    def showPaths(self,paths = 10):
        N = 20000  # time steps
        T = self.T
        T_vec, dt = np.linspace(0, T, N, retstep=True)

       
        std_asy = np.sqrt(sigma**2 / (2 * kappa))  # asymptotic standard deviation

        X0 = self.r0
        X = np.zeros((N, paths))
        X[0, :] = X0
        W = ss.norm.rvs(loc=0, scale=1, size=(N - 1, paths))

        # Uncomment for Euler Maruyama
        # for t in range(0,N-1):
        #    X[t + 1, :] = X[t, :] + kappa*(theta - X[t, :])*dt + sigma * np.sqrt(dt) * W[t, :]

        std_dt = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))
        for t in range(0, N - 1):
            X[t + 1, :] = theta + np.exp(-kappa * dt) * (X[t, :] - theta) + std_dt * W[t, :]

        # X_T = X[-1, :]  # values of X at time T
        # X_1 = X[:, 1]  # a single path

        # mean_T = theta + np.exp(-kappa * T) * (X0 - theta)
        # std_T = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * T)))
        
        # x = np.linspace(X_T.min(), X_T.max(), 100)
        fig = plt.figure(figsize=(16, 5))
        ax1 = fig.add_subplot(111)
        ax1.plot(T_vec, X[:, :paths], linewidth=0.5)
        ax1.plot(T_vec, (theta + std_asy) * np.ones_like(T_vec), label="1 asymptotic std dev", color="black")
        ax1.plot(T_vec, (theta - std_asy) * np.ones_like(T_vec), color="black")
        ax1.plot(T_vec, theta * np.ones_like(T_vec), label="Long term mean")
        ax1.legend(loc="upper right")
        ax1.set_title(f"{paths} OU processes")
        ax1.set_xlabel("T")
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
        ax.set_title(f'Vasicek Model bond Price Surface')
        
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
        plt.title("Predicted Term Structure from the Vasicek Model")
        plt.grid(True)
        plt.legend()
        plt.show()

    
  

class OU_process:
    """
    Class for the OU process:
    theta = long term mean
    sigma = diffusion coefficient
    kappa = mean reversion coefficient
    """

    def __init__(self, sigma=0.2, theta=-0.1, kappa=0.1):
        self.theta = theta
        if sigma < 0 or kappa < 0:
            raise ValueError("sigma,theta,kappa must be positive")
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

        std_dt = np.sqrt(self.sigma**2 / (2 * self.kappa) * (1 - np.exp(-2 * self.kappa * dt)))
        for t in range(0, N - 1):
            X[t + 1, :] = self.theta + np.exp(-self.kappa * dt) * (X[t, :] - self.theta) + std_dt * W[t, :]

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

    r0 = 0.05
    T = 5.0
    sigma = 0.15
    theta = 0.02
    kappa = 3.0

    bond_info = Bond_info(r0,T)
    process_info = Process_info(sigma,theta,kappa,OU_process(sigma,theta,kappa))

    rates = [0.01, 3]
    maturities = [0.01, 5]

    pricer = Vasicek_pricer(bond_info,process_info) 
    print(pricer.closed_formula())
    # pricer.priceSurface(rates,maturities)
    pricer.showPaths()
    # pricer.plotTermStructure()
    
    
    
        