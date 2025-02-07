import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import pandas as pd
from time import time

# ============================================
#
# 	            Hull-White Model
#
#   dr_t = θ(t)dt + σdW_t
#
# where:
#   dr_t = k(θ(t) - r_t)dt + σdW_t
#
# Key Parameters:
#   ​1. 𝜅: Mean Reversion Speed
#   2. θ(t): Time-Dependent Mean Reversion Level
#   3. σ: Volatility of Short Rate
#
# ============================================

class HullWhite_Pricer:
    """
    Closed-Form solution
    Monte Carlo
    Fourier techniques (FFT, IF) 
    """
    def __init__(self, Asset_Info, Process_Info):
        self.kappa = Process_Info.kappa
        self.theta_t = Process_Info.theta_t
        self.sigma = Process_Info.sigma

        self.underlyingClass = Asset_Info.underlyingClass
        self.termStructure = Asset_Info.termStructure # pd df of the TS where columns are Tenor and Spot rates
        self.maturities = self.termStructure["maturities"].to_numpy()
        self.yields = self.termStructure["yields"].to_numpy()
        self.t = Asset_Info.t
        self.T = Asset_Info.T   


    def closed_formula(self):
        """
        Closed-form solution of Zero-Coupon Bond
        """
        def B(t, T):
            """
            Sensitivity to short rate
            """
            return (1 - np.exp(-self.kappa * (T - t))) / self.kappa

        def f():
            """
            instantaneous forward rate f(0,s) from a given yield curve
            """
             
            dy_ds = np.diff(self.yields) / np.diff(self.maturities)  
            dy_ds = np.append(dy_ds, (self.yields[-1] - self.yields[-2]) / (self.maturities[-1] - self.maturities[-2]))  
            
            forward_rates = self.yields + self.maturities * dy_ds
            
            return forward_rates
        
        def A(t, T):
            """
            Ensures the bond price is arbitrage-free and fits the initial yield curve
            """
            forward_rates = f()
            forward_rate_at_t = np.interp(T, self.maturities, forward_rates)
            
            return (forward_rate_at_t - self.sigma**2 / (2 * self.kappa**2)) * (T - t)
        
        A_t_T = A(self.t, self.T)
        B_t_T = B(self.t, self.T)
        
        bond_price = np.exp(-A_t_T - (self.sigma**2 * B_t_T**2) / (2 * self.kappa) * (self.T - self.t))
        
        return bond_price
    
    

class Asset_Info:
    def __init__(self,underlyingClass,termStructure,t,T):
        self.underlyingClass = underlyingClass
        self.termStructure = termStructure
        self.t = t
        self.T = T

class Process_Info:
    def __init__(self,kappa,theta_t,sigma):
        self.kappa = kappa
        self.theta_t = theta_t
        self.sigma = sigma


if __name__ == "__main__":
    
    maturities = [1/12, 2/12, 3/12, 4/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]  
    yields = [0.0437, 0.0437, 0.0431, 0.0433, 0.0428, 0.0417, 0.0422, 0.0427, 0.0436, 0.0447, 0.0458, 0.0488, 0.0483]  
    yieldCurve = pd.DataFrame({'maturities': maturities, 'yields': yields})

    underlyingClass = "IRSwap"
    termStructure = yieldCurve
    t = 0
    T = 5 
    kappa = 3
    theta_t = 2
    sigma = 0.3


    assetInfo = Asset_Info(underlyingClass,termStructure,t,T)
    processInfo = Process_Info(kappa,theta_t,sigma)

    pricer = HullWhite_Pricer(assetInfo,processInfo)
    
    start = time()
    price = pricer.closed_formula()
    end = time() - start
    print(f"price of ZCB: {price:.4f}")
    print(f"computation time: {end:.10f}")

    kappa = 0.03  # Mean reversion speed
    sigma = 0.01  # Volatility
    t = 0  # Current time

    # Compute the instantaneous forward rates
    dy_ds = np.diff(yields) / np.diff(maturities)
    dy_ds = np.append(dy_ds, (yields[-1] - yields[-2]) / (maturities[-1] - maturities[-2]))

    forward_rates = yields + maturities * dy_ds

    # Interpolate the yield curve for smoother plotting
    smooth_maturities = np.linspace(min(maturities), max(maturities), 100)
    smooth_yields = np.interp(smooth_maturities, maturities, yields)
    smooth_forward = np.interp(smooth_maturities, maturities, forward_rates)

    # Plot yield curve and instantaneous forward rates
    plt.figure(figsize=(10, 5))
    plt.plot(smooth_maturities, smooth_yields, label="Yield Curve", linestyle='-')
    plt.plot(smooth_maturities, smooth_forward, label="Instantaneous Forward Curve", linestyle='-')

    # Labels and legend
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Rate")
    plt.title("Hull-White Yield Curve and Instantaneous Forward Rate")
    plt.legend()
    plt.grid()

    # Show plot
    plt.show()


    
        
        




        