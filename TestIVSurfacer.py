import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import pandas as pd
import yfinance as yf
import datetime as dt
N =norm.cdf
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import time




def BS_calls(S,K,T,r,sigma):
    d1 = np.log(S/K) + (r + sigma**2/2)*T / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*N(d1) - K * np.exp(-r*T)*N(d2)


def BS_puts(S,K,T,r,sigma):
    d1 = np.log(S/K) + (r + sigma**2/2)*T / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r*T)*N(-d2) - S*N(-d1) 

def implied_vol(mkt_val,S,K,T,r,option_type):
    def callObj(sigma):
        return abs(BS_calls(S,K,T,r,sigma) - mkt_val)
    
    def putObj(sigma):
        return abs(BS_puts(S,K,T,r,sigma) - mkt_val)
    
    if option_type == 'call':
        res = minimize_scalar(callObj,bounds=(0.0001,6), method='bounded')
        return res.x
    elif option_type == 'put':
        res = minimize_scalar(putObj,bounds=(0.0001,6), method='bounded')
        return res.x
    else:
        raise ValueError("option_type must be either a 'call' or 'put' ")
    

if __name__ == "__main__":
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.45

    c = BS_calls(S,K,T,r,sigma)
    iv = implied_vol(c,S,K,T,r,'call')
    print(round(iv,4))

    