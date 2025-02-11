import numpy as np
from scipy.fftpack import ifft
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.integrate import quad
import scipy.special as scps
from math import factorial
from functools import partial

def cf_Heston_good(u, t, v0, mu, kappa, theta, sigma, rho):
    """
    Heston characteristic function as proposed by Schoutens (2004)
    """
    xi = kappa - sigma * rho * u * 1j
    d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
    g1 = (xi + d) / (xi - d)
    g2 = 1 / g1
    cf = np.exp(
        1j * u * mu * t
        + (kappa * theta) / (sigma**2) * ((xi - d) * t - 2 * np.log((1 - g2 * np.exp(-d * t)) / (1 - g2)))
        + (v0 / sigma**2) * (xi - d) * (1 - np.exp(-d * t)) / (1 - g2 * np.exp(-d * t))
    )
    return cf

def fft_Lewis(K, S0, r, T, cf, interp="cubic"):
    """
    K = vector of strike
    S = spot price scalar
    cf = characteristic function
    interp can be cubic or linear
    """
    N = 2**15  # FFT more efficient for N power of 2
    B = 500  # integration limit
    dx = B / N
    x = np.arange(N) * dx  # the final value B is excluded

    weight = np.arange(N)  # Simpson weights
    weight = 3 + (-1) ** (weight + 1)
    weight[0] = 1
    weight[N - 1] = 1

    dk = 2 * np.pi / B
    b = N * dk / 2
    ks = -b + dk * np.arange(N)

    integrand = np.exp(-1j * b * np.arange(N) * dx) * cf(x - 0.5j) * 1 / (x**2 + 0.25) * weight * dx / 3
    integral_value = np.real(ifft(integrand) * N)

    if interp == "linear":
        spline_lin = interp1d(ks, integral_value, kind="linear")
        prices = S0 - np.sqrt(S0 * K) * np.exp(-r * T) / np.pi * spline_lin(np.log(S0 / K))
    elif interp == "cubic":
        spline_cub = interp1d(ks, integral_value, kind="cubic")
        prices = S0 - np.sqrt(S0 * K) * np.exp(-r * T) / np.pi * spline_cub(np.log(S0 / K))
    return prices

def IV_from_Lewis(K, S0, T, r, cf, disp=False):
    """Implied Volatility from the Lewis formula
    K = strike; S0 = spot stock; T = time to maturity; r = interest rate
    cf = characteristic function"""
    k = np.log(S0 / K)

    def obj_fun(sig):
        integrand = (
            lambda u: np.real(
                np.exp(u * k * 1j)
                * (cf(u - 0.5j) - np.exp(1j * u * r * T + 0.5 * r * T) * np.exp(-0.5 * T * (u**2 + 0.25) * sig**2))
            )
            * 1
            / (u**2 + 0.25)
        )
        int_value = quad(integrand, 1e-15, 500, limit=2000, full_output=1)[0]
        return int_value

    # X0 = [0.2, 1, 2, 4, 0.0001]  # set of initial guess points
    X0 = [0.1, 0.2, 0.3, 0.5]  # set of initial guess points
    for x0 in X0:
        x, _, solved, msg = fsolve(
            obj_fun,
            [
                x0,
            ],
            full_output=True,
            xtol=1e-4,
        )
        if solved == 1:
            return x[0]
    if disp is True:
        print("Strike", K, msg)
    return -1

def Q1(k, cf, right_lim):
    """
    P(X<k) - Probability to be in the money under the stock numeraire.
    cf: characteristic function
    right_lim: right limit of integration
    """

    def integrand(u):
        return np.real((np.exp(-u * k * 1j) / (u * 1j)) * cf(u - 1j) / cf(-1.0000000000001j))

    return 1 / 2 + 1 / np.pi * quad(integrand, 1e-15, right_lim, limit=5000)[0]

def Q2(k, cf, right_lim):
    """
    P(X<k) - Probability to be in the money under the money market numeraire
    cf: characteristic function
    right_lim: right limit of integration
    """

    def integrand(u):
        return np.real(np.exp(-u * k * 1j) / (u * 1j) * cf(u))

    return 1 / 2 + 1 / np.pi * quad(integrand, 1e-15, right_lim, limit=5000)[0]

def Gil_Pelaez_pdf(x, cf, right_lim):
    """
    Gil Pelaez formula for the inversion of the characteristic function
    INPUT
    - x: is a number
    - right_lim: is the right extreme of integration
    - cf: is the characteristic function
    OUTPUT
    - the value of the density at x.
    """

    def integrand(u):
        return np.real(np.exp(-u * x * 1j) * cf(u))

    return 1 / np.pi * quad(integrand, 1e-15, right_lim)[0]

def Heston_pdf(i, t, v0, mu, theta, sigma, kappa, rho):
    """
    Heston density by Fourier inversion.
    """
    cf_H_b_good = partial(
        cf_Heston_good,
        t=t,
        v0=v0,
        mu=mu,
        theta=theta,
        sigma=sigma,
        kappa=kappa,
        rho=rho,
    )
    return Gil_Pelaez_pdf(i, cf_H_b_good, np.inf)

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

def Merton_pdf(x, T, mu, sig, lam, muJ, sigJ):
    """
    Merton density function
    """
    tot = 0
    for k in range(20):
        tot += (
            (lam * T) ** k
            * np.exp(-((x - mu * T - k * muJ) ** 2) / (2 * (T * sig**2 + k * sigJ**2)))
            / (factorial(k) * np.sqrt(2 * np.pi * (sig**2 * T + k * sigJ**2)))
        )
    return np.exp(-lam * T) * tot

def NIG_pdf(x, T, c, theta, sigma, kappa):
    """
    Merton density function
    """
    A = theta / (sigma**2)
    B = np.sqrt(theta**2 + sigma**2 / kappa) / sigma**2
    C = T / np.pi * np.exp(T / kappa) * np.sqrt(theta**2 / (kappa * sigma**2) + 1 / kappa**2)
    return (
        C
        * np.exp(A * (x - c * T))
        * scps.kv(1, B * np.sqrt((x - c * T) ** 2 + T**2 * sigma**2 / kappa))
        / np.sqrt((x - c * T) ** 2 + T**2 * sigma**2 / kappa)
    )
