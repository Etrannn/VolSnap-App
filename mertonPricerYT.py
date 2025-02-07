import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from functools import partial
from scipy.optimize import minimize
from math import factorial
from scipy.integrate import quad



mu = 0.05  # Drift term for the Brownian motion (average return)
sig = 0.2  # Volatility of the Brownian motion (how much the asset fluctuates)
lam = 1.2  # Jump intensity (average number of jumps per unit time)
muJ = 0.15  # Mean size of each jump
sigJ = 0.5  # Standard deviation of the jump size
T = 2  # Time horizon for the simulation
N = 1000000  # Number of paths (random samples) to simulate
# Simulating the Process
np.random.seed(seed=42)
W = ss.norm.rvs(0, 1, N)  # The normal RV vector
# Generates N random Poisson variables with mean λT, representing the number of jumps for each path
P = ss.poisson.rvs(lam * T, size=N)  # Poisson random vector
Jumps = np.asarray([ss.norm.rvs(muJ, sigJ, i).sum() for i in P])  # Jumps vector
X_T = mu * T + np.sqrt(T) * sig * W + Jumps  # Merton process
# Merton Density Function
def Merton_density(x, T, mu, sig, lam, muJ, sigJ):
    tot = 0
    for k in range(20):
        # Accumulates the summation term up to k=20 (approximation of the infinite sum)
        tot += (
            (lam * T) ** k
            * np.exp(-((x - mu * T - k * muJ) ** 2) / (2 * (T * sig**2 + k * sigJ**2)))
            / (factorial(k) * np.sqrt(2 * np.pi * (sig**2 * T + k * sigJ**2)))
        )
    return np.exp(-lam * T) * tot
# Characteristic Function
def cf_mert(u, t=1, mu=1, sig=2, lam=0.8, muJ=0, sigJ=0.5):
    """
    Characteristic function of a Merton random variable at time t
    mu: drift
    sig: diffusion coefficient
    lam: jump activity
    muJ: jump mean size
    sigJ: jump size standard deviation
    1.	Defines the characteristic function of the Merton process.
    2.	Formula matches the theoretical CF for jump-diffusion processes.
    """
    return np.exp(
        t * (1j * u * mu - 0.5 * u**2 * sig**2 + lam * (np.exp(1j * u * muJ - 0.5 * u**2 * sigJ**2) - 1))
    )
# Gil-Pelaez PDF Inversion
def Gil_Pelaez_pdf(x, cf, right_lim):
    """
    Gil Pelaez formula for the inversion of the characteristic function
    INPUT
    - x: is a number
    - right_lim: is the right extreme of integration
    - cf: is the characteristic function
    OUTPUT
    - the value of the density at x.
    1.	Computes the PDF by inverting the characteristic function using the Gil-Pelaez theorem.
    2.	quad: Performs numerical integration from 00 to right_lim.
    """
    def integrand(u):
        return np.real(np.exp(-u * x * 1j) * cf(u))
    return 1 / np.pi * quad(integrand, 1e-15, right_lim)[0]
# Simulation Plot
cf_M_b = partial(cf_mert, t=T, mu=mu, sig=sig, lam=lam, muJ=muJ, sigJ=sigJ)
x = np.linspace(X_T.min(), X_T.max(), 500)
y = np.linspace(-3, 5, 50)
'''
1.	Merton_density: Plots the theoretical density.
2.	Gil_Pelaez_pdf: Plots the PDF derived from the CF.
3.	plt.hist: Overlays a histogram of simulated values X_T.
4.	qqplot: Visualizes the quantiles of X_T compared to a normal distribution.
'''
plt.figure(figsize=(15, 6))
plt.plot(x, Merton_density(x, T, mu, sig, lam, muJ, sigJ), color="r", label="Merton density")
plt.plot(y, [Gil_Pelaez_pdf(i, cf_M_b, np.inf) for i in y], "p", label="Fourier inversion")
plt.hist(X_T, density=True, bins=200, facecolor="LightBlue", label="frequencies of X_T")
plt.legend()
plt.title("Merton Histogram")
plt.show()
qqplot(X_T, line="s")
plt.show()
print(f"Median: {np.median(X_T)}")
print(f"Mean: {np.mean(X_T)}")
print(f"Standard Deviation: {np.std(X_T)}")
print(f"Skewness: {ss.skew(X_T)}")
print(f"Kurtosis: {ss.kurtosis(X_T)}")
print(f"Calculated average growth rate (per unit time): {np.mean(X_T) / T}")
print(f"Theoretical average growth rate (per unit time): {mu + lam * muJ}")
print("\n\n\n")
# Maximum Likelihood Estimation
# log_likely_Merton: Computes the negative log-likelihood of the Merton density for parameter estimation.
def log_likely_Merton(x, data, T):
    return (-1) * np.sum(np.log(Merton_density(data, T, x[0], x[1], x[2], x[3], x[4])))

# minimize: Finds the parameters that maximize the likelihood using BFGS.
result_Mert = minimize(
    log_likely_Merton, x0=[0.1, 0.5, 1, 0.1, 1], method="BFGS", args=(X_T, T)
)  # Try also Nelder-Mead
# Displays the results of the optimization and compares original and estimated parameters.
print("Number of iterations performed by the optimizer: ", result_Mert.nit)
print(f"Original parameters: mu={mu}, sigma={sig}, lam={lam}, alpha={muJ}, xi={sigJ} ")
print("MLE parameters: ", result_Mert.x)