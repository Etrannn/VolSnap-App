import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
# from blackScholesPricer import BS_pricer, Option_param, process_info,Diffusion_process

class BlackScholesMCPricer:
    """
    A numerical solution to derive a European option's price, assuming the underlying asset follows a GBM.

    Parameters:
    - S: Stock price at time 0
    - K: Option strike price
    - r: Risk-free rate at time 0 (annualized in decimal format)
    - T: Time to maturity of the option contract
    - sigma: Asset's volatility (annualized in decimal format)
    - option_type: Specifies the user's option type ('call' or 'put')
    - N: Number of time steps
    - M: Number of simulations
    - q: Continuous dividend yield (annualized, decimal, default 0) 
    - antithetic_reduction: A method to reduce the variance of results from a Monte Carlo simulation (On by default)
    - delta_control: A method to reduce the variance of results from a Monte Carlo simulation (On by default)
    - gamma_control: A method to reduce the variance of results from a Monte Carlo simulation (On by default)

    Output: 
    European option price and standard error and ability to visualise MC distribution

    Variance reduction feature: User has ability to individually turn the three reduction techniques On or have antithetic reduction and delta_control On or 
    have all the variance techniques On.
    """
    def __init__(self, S:float, K:float, r:float, T:float, sigma:float, option_type='call', N=365, M=1000, q=0, antithetic_reduction=True, delta_control=True, gamma_control=True):
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.option_type = option_type
        self.N = N
        self.M = M
        self.q = q
        self.antithetic_reduction = antithetic_reduction
        self.delta_control = delta_control
        self.gamma_control = gamma_control

    def monteCarlo(self):
        # Precompute constants
        dt = self.T / self.N
        nudt = (self.r - self.q - 0.5 * self.sigma ** 2) * dt
        volsdt = self.sigma * np.sqrt(dt)
        lnS = np.log(self.S)

        Z = np.random.normal(size=(self.N, self.M))



        if self.antithetic_reduction == False and self.delta_control == False and self.gamma_control == False:
            """No MC control variates/reduction are active"""
            # Simulate price paths using the Monte Carlo method
            
            delta_lnSt = nudt + volsdt * Z
            lnSt = lnS + np.cumsum(delta_lnSt, axis=0)
            lnSt = np.concatenate((np.full(shape=(1, self.M), fill_value=lnS), lnSt))

            ST = np.exp(lnSt)
            
            if self.option_type == 'call':
                CT = np.maximum(0, ST[-1] - self.K)
                C0 = np.exp(-self.r * self.T) * np.mean(CT)
                StdErr = self._std_error(CT, C0)
                return C0, StdErr

            elif self.option_type == 'put':
                PT = np.maximum(0, self.K - ST[-1])
                P0 = np.exp(-self.r * self.T) * np.mean(PT)
                StdErr = self._std_error(PT, P0)
                return P0, StdErr

            else:
                raise ValueError("Invalid option type. Must be 'call' or 'put'.")

        elif self.antithetic_reduction == True and self.delta_control == False and self.gamma_control == False:
            """Antithetic reduction is On"""

            # Monte Carlo Method
            delta_lnSt1 = nudt + volsdt*Z
            delta_lnSt2 = nudt - volsdt*Z
            lnSt1 = lnS + np.cumsum(delta_lnSt1, axis=0)
            lnSt2 = lnS + np.cumsum(delta_lnSt2, axis=0)

            # Compute Expectation and SE
            ST1 = np.exp(lnSt1)
            ST2 = np.exp(lnSt2)

            if self.option_type == "call":
                CT = 0.5 * ( np.maximum(0, ST1[-1] - self.K) + np.maximum(0, ST2[-1] - self.K) )
                C0_av = np.exp(-self.r*self.T)*np.sum(CT)/self.M

                sigma_av = np.sqrt( np.sum( (np.exp(-self.r*self.T)*CT - C0_av)**2) / (self.M-1) )
                SE_av = sigma_av/np.sqrt(self.M)
                return C0_av, SE_av
            
            elif self.option_type == "put":
                PT = 0.5 * ( np.maximum(0, self.K - ST1[-1]) + np.maximum(0, self.K - ST2[-1]) )
                P0_av = np.exp(-self.r*self.T)*np.sum(PT)/self.M

                sigma_av = np.sqrt( np.sum( (np.exp(-self.r*self.T)*PT - P0_av)**2) / (self.M-1) )
                SE_av = sigma_av/np.sqrt(self.M)
                return P0_av, SE_av
            
            else:
                raise ValueError("Invalid option type. Must be 'call' or 'put'.")
    
        elif self.antithetic_reduction == False and self.delta_control == True and self.gamma_control == False:
            """Delta control variate is On"""
            erdt = np.exp(self.r*dt)
            cv = 0
            beta1 = -1

            # Monte Carlo Method
            delta_St = nudt + volsdt*Z
            ST = self.S*np.cumprod( np.exp(delta_St), axis=0)
            ST = np.concatenate( (np.full(shape=(1, self.M), fill_value=self.S), ST ) )
            
            if self.option_type == "call":
                deltaSt = self.delta_calc(self.r, ST[:-1].T, self.K, np.linspace(self.T,dt,self.N), self.sigma, "call").T
                cv = np.cumsum(deltaSt*(ST[1:] - ST[:-1]*erdt), axis=0)

                CT = np.maximum(0, ST[-1] - self.K) + beta1*cv[-1]
                C0_dv = np.exp(-self.r*self.T)*np.sum(CT)/self.M

                sigma_dv = np.sqrt( np.sum( (np.exp(-self.r*self.T)*CT - C0_dv)**2) / (self.M-1) )
                sigma_dv = np.std(np.exp(-self.r*self.T)*CT)
                SE_dv = sigma_dv/np.sqrt(self.M)
                return C0_dv, SE_dv
            
            elif self.option_type == "put":
                deltaSt = self.delta_calc(self.r, ST[:-1].T, self.K, np.linspace(self.T,dt,self.N), self.sigma, "put").T
                cv = np.cumsum(deltaSt*(ST[1:] - ST[:-1]*erdt), axis=0)

                PT = np.maximum(0, self.K - ST[-1]) + beta1*cv[-1]
                P0_dv = np.exp(-self.r*self.T)*np.sum(PT)/self.M

                sigma_dv = np.sqrt( np.sum( (np.exp(-self.r*self.T)*PT - P0_dv)**2) / (self.M-1) )
                sigma_dv = np.std(np.exp(-self.r*self.T)*PT)
                SE_dv = sigma_dv/np.sqrt(self.M)
                return P0_dv, SE_dv
            else:
                raise ValueError("Invalid option type. Must be 'call' or 'put'.")

        elif self.antithetic_reduction == False and self.delta_control == False and self.gamma_control == True:
            """gamma control variate is On"""
            erdt = np.exp(self.r*dt)
            ergamma = np.exp((2*self.r+self.sigma**2)*dt) - 2*erdt + 1
            beta2 = -0.5

            # Monte Carlo Method
            delta_St = nudt + volsdt*Z
            ST = self.S*np.cumprod( np.exp(delta_St), axis=0)
            ST = np.concatenate( (np.full(shape=(1, self.M), fill_value=self.S), ST ) )
            gammaSt = self.gamma_calc(self.r, ST[:-1].T, self.K, np.linspace(self.T,dt,self.N), self.sigma).T
            cv2 = np.cumsum(gammaSt*((ST[1:] - ST[:-1])**2 - ergamma*ST[:-1]**2), axis=0)

            if self.option_type == "call":
                CT = np.maximum(0, ST[-1] - self.K) + beta2*cv2[-1]
                C0_gv = np.exp(-self.r*self.T)*np.sum(CT)/self.M

                sigma = np.sqrt( np.sum( (np.exp(-r*T)*CT - C0_gv)**2) / (self.M-1) )
                sigma = np.std(np.exp(-self.r*self.T)*CT)
                SE_gv = sigma/np.sqrt(self.M)
                return C0_gv, SE_gv
            
            elif self.option_type == "put":
                PT = np.maximum(0, ST[-1] - self.K) + beta2*cv2[-1]
                P0_gv = np.exp(-self.r*self.T)*np.sum(PT)/self.M

                sigma = np.sqrt( np.sum( (np.exp(-self.r*self.T)*PT - P0_gv)**2) / (self.M-1) )
                sigma = np.std(np.exp(-self.r*self.T)*PT)
                SE_gv = sigma/np.sqrt(self.M)
                return P0_gv, SE_gv
    
            else:
                raise ValueError("Invalid option type. Must be 'call' or 'put'.")

        
        elif self.antithetic_reduction == True and self.delta_control == True and self.gamma_control == False:
            """Antithetic reduction and delta control variate is On"""
            erdt = np.exp(self.r*dt)
            beta1 = -1

            # Monte Carlo Method
            delta_St1 = nudt + volsdt*Z
            delta_St2 = nudt - volsdt*Z
            ST1 = self.S*np.cumprod( np.exp(delta_St1), axis=0)
            ST2 = self.S*np.cumprod( np.exp(delta_St2), axis=0)
            ST1 = np.concatenate( (np.full(shape=(1, self.M), fill_value=self.S), ST1 ) )
            ST2 = np.concatenate( (np.full(shape=(1, self.M), fill_value=self.S), ST2 ) )

            # Calculate delta for both sets of underlying stock prices
            deltaSt1 = self.delta_calc(self.r, ST1[:-1].T, self.K, np.linspace(self.T,dt,self.N), self.sigma, "call").T
            deltaSt2 = self.delta_calc(self.r, ST2[:-1].T, self.K, np.linspace(self.T,dt,self.N), self.sigma, "call").T

            # Calculate two sets of delta control variates for negatively correlated assets
            cv11 = np.cumsum(deltaSt1*(ST1[1:] - ST1[:-1]*erdt), axis=0)
            cv12 = np.cumsum(deltaSt2*(ST2[1:] - ST2[:-1]*erdt), axis=0)

            if self.option_type == "call":

                deltaSt1 = self.delta_calc(self.r, ST1[:-1].T, self.K, np.linspace(self.T,dt,self.N), self.sigma, "call").T
                deltaSt2 = self.delta_calc(self.r, ST2[:-1].T, self.K, np.linspace(self.T,dt,self.N), self.sigma, "call").T

                # Calculate two sets of delta control variates for negatively correlated assets
                cv11 = np.cumsum(deltaSt1*(ST1[1:] - ST1[:-1]*erdt), axis=0)
                cv12 = np.cumsum(deltaSt2*(ST2[1:] - ST2[:-1]*erdt), axis=0)

                CT = 0.5 * (  np.maximum(0, ST1[-1] - self.K) + beta1*cv11[-1]
                            + np.maximum(0, ST2[-1] - self.K) + beta1*cv12[-1] )

                C0_adv = np.exp(-self.r*self.T)*np.sum(CT)/self.M

                sigma = np.sqrt( np.sum( (np.exp(-self.r*self.T)*CT - C0_adv)**2) / (self.M-1) )
                sigma = np.std(np.exp(-self.r*self.T)*CT)
                SE_adv = sigma/np.sqrt(self.M)
                return C0_adv, SE_adv

            elif self.option_type == "put":
                
                deltaSt1 = self.delta_calc(self.r, ST1[:-1].T, self.K, np.linspace(self.T,dt,self.N), self.sigma, "put").T
                deltaSt2 = self.delta_calc(self.r, ST2[:-1].T, self.K, np.linspace(self.T,dt,self.N), self.sigma, "put").T

                # Calculate two sets of delta control variates for negatively correlated assets
                cv11 = np.cumsum(deltaSt1*(ST1[1:] - ST1[:-1]*erdt), axis=0)
                cv12 = np.cumsum(deltaSt2*(ST2[1:] - ST2[:-1]*erdt), axis=0)

                PT = 0.5 * (  np.maximum(0, self.K - ST1[-1]) + beta1*cv11[-1]
                            + np.maximum(0, self.K - ST2[-1]) + beta1*cv12[-1] )

                P0_adv = np.exp(-self.r*self.T)*np.sum(PT)/self.M

                sigma = np.sqrt( np.sum( (np.exp(-self.r*self.T)*PT - P0_adv)**2) / (self.M-1) )
                sigma = np.std(np.exp(-self.r*self.T)*PT)
                SE_adv = sigma/np.sqrt(self.M)
                return P0_adv, SE_adv
            else:
                raise ValueError("Invalid option type. Must be 'call' or 'put'.")

                
        elif self.antithetic_reduction == True and self.delta_control == True and self.gamma_control == True:
            """Antithetic reduction, delta control and gamma control variates are all On."""
            erdt = np.exp(self.r*dt)
            ergamma = np.exp((2*self.r+self.sigma**2)*dt) - 2*erdt + 1

            beta1 = -1
            beta2 = -0.5

            # Monte Carlo Method
            delta_St1 = nudt + volsdt*Z
            delta_St2 = nudt - volsdt*Z
            ST1 = self.S*np.cumprod( np.exp(delta_St1), axis=0)
            ST2 = self.S*np.cumprod( np.exp(delta_St2), axis=0)
            ST1 = np.concatenate( (np.full(shape=(1, self.M), fill_value=self.S), ST1 ) )
            ST2 = np.concatenate( (np.full(shape=(1, self.M), fill_value=self.S), ST2 ) )

            if self.option_type == "call":

                # Calculate delta for both sets of underlying stock prices
                deltaSt1 = self.delta_calc(self.r, ST1[:-1].T, self.K, np.linspace(self.T,dt,self.N), self.sigma, "call").T
                deltaSt2 = self.delta_calc(self.r, ST2[:-1].T, self.K, np.linspace(self.T,dt,self.N), self.sigma, "call").T

                # Calculate gamma for both sets of underlying stock prices
                gammaSt1 = self.gamma_calc(self.r, ST1[:-1].T, self.K, np.linspace(self.T,dt,self.N), self.sigma).T
                gammaSt2 = self.gamma_calc(self.r, ST2[:-1].T, self.K, np.linspace(self.T,dt,self.N), self.sigma).T

                # Calculate two sets of delta control variates for negatively correlated assets
                cv11 = np.cumsum(deltaSt1*(ST1[1:] - ST1[:-1]*erdt), axis=0)
                cv12 = np.cumsum(deltaSt2*(ST2[1:] - ST2[:-1]*erdt), axis=0)

                # Calculate two sets of gamma control variates for negatively correlated assets
                cv21 = np.cumsum(gammaSt1*((ST1[1:] - ST1[:-1])**2 - ergamma*ST1[:-1]**2), axis=0)
                cv22 = np.cumsum(gammaSt2*((ST2[1:] - ST2[:-1])**2 - ergamma*ST2[:-1]**2), axis=0)

                CT = 0.5 * (  np.maximum(0, ST1[-1] - self.K) + beta1*cv11[-1] + beta2*cv21[-1]
                            + np.maximum(0, ST2[-1] - self.K) + beta1*cv12[-1] + beta2*cv22[-1])

                C0_adgv = np.exp(-self.r*self.T)*np.sum(CT)/self.M

                sigma = np.sqrt( np.sum( (np.exp(-self.r*self.T)*CT - C0_adgv)**2) / (self.M-1) )
                sigma = np.std(np.exp(-self.r*self.T)*CT)
                SE_adgv = sigma/np.sqrt(self.M)
                return C0_adgv, SE_adgv
            
            elif self.option_type == "put":
                
                # Calculate delta for both sets of underlying stock prices
                deltaSt1 = self.delta_calc(self.r, ST1[:-1].T, self.K, np.linspace(self.T,dt,self.N), self.sigma, "put").T
                deltaSt2 = self.delta_calc(self.r, ST2[:-1].T, self.K, np.linspace(self.T,dt,self.N), self.sigma, "put").T

                # Calculate gamma for both sets of underlying stock prices
                gammaSt1 = self.gamma_calc(self.r, ST1[:-1].T, self.K, np.linspace(self.T,dt,self.N), self.sigma).T
                gammaSt2 = self.gamma_calc(self.r, ST2[:-1].T, self.K, np.linspace(self.T,dt,self.N), self.sigma).T

                # Calculate two sets of delta control variates for negatively correlated assets
                cv11 = np.cumsum(deltaSt1*(ST1[1:] - ST1[:-1]*erdt), axis=0)
                cv12 = np.cumsum(deltaSt2*(ST2[1:] - ST2[:-1]*erdt), axis=0)

                # Calculate two sets of gamma control variates for negatively correlated assets
                cv21 = np.cumsum(gammaSt1*((ST1[1:] - ST1[:-1])**2 - ergamma*ST1[:-1]**2), axis=0)
                cv22 = np.cumsum(gammaSt2*((ST2[1:] - ST2[:-1])**2 - ergamma*ST2[:-1]**2), axis=0)

                PT = 0.5 * (  np.maximum(0, ST1[-1] - self.K) + beta1*cv11[-1] + beta2*cv21[-1]
                            + np.maximum(0, ST2[-1] - self.K) + beta1*cv12[-1] + beta2*cv22[-1])

                P0_adgv = np.exp(-self.r*self.T)*np.sum(PT)/self.M

                sigma = np.sqrt( np.sum( (np.exp(-self.r*self.T)*PT - P0_adgv)**2) / (self.M-1) )
                sigma = np.std(np.exp(-self.r*self.T)*PT)
                SE_adgv = sigma/np.sqrt(self.M)
                return P0_adgv, SE_adgv
            
            else:
                raise ValueError("Invalid option type. Must be 'call' or 'put'.")
        else:
            raise ValueError("Please set variance reduction techniques to be either: \n - Antithetic reduction Only \n - Delta Control Only \n - Gamma Control only \n - Antithetic reduction and Delta control \n - all techniques on.")


    def _std_error(self, last_option_price, final_option_price):
        sigma = np.std(last_option_price)
        SE = sigma / np.sqrt(self.M)
        return SE
    
    def visualDist(self,market_value):
        """
        Visualises MC's estimated probability distribution to a given market price of the market
        Parameters:
        - market_value: The price of the measured option ($)

        Output:
        A visual of the estimated probability distribution compared to the market value of the option
        """
        optionVal0,SE = self.monteCarlo()

        x1 = np.linspace(optionVal0-3*SE, optionVal0-1*SE, 100)
        x2 = np.linspace(optionVal0-1*SE, optionVal0+1*SE, 100)
        x3 = np.linspace(optionVal0+1*SE, optionVal0+3*SE, 100)

        s1 = stats.norm.pdf(x1, optionVal0, SE)
        s2 = stats.norm.pdf(x2, optionVal0, SE)
        s3 = stats.norm.pdf(x3, optionVal0, SE)

        plt.fill_between(x1, s1, color='tab:blue',label='> StDev')
        plt.fill_between(x2, s2, color='cornflowerblue',label='1 StDev')
        plt.fill_between(x3, s3, color='tab:blue')

        plt.plot([optionVal0,optionVal0],[0, max(s2)*1.1], 'k',
                label='Monte Carlo Value')
        plt.plot([market_value,market_value],[0, max(s2)*1.1], 'r',
                label='Closed Solution Value')

        plt.ylabel("Probability")
        plt.xlabel("Option Price")
        plt.legend()
        plt.show()
    
    def delta_calc(self, r, S, K, T, sigma, type="call"):
        """Calculate delta of an option"""
        d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
        try:
            if type == "call":
                delta_calc = stats.norm.cdf(d1, 0, 1)
            elif type == "put":
                delta_calc = -stats.norm.cdf(-d1, 0, 1)
            return delta_calc
        except:
            print("Please confirm option type, either 'call' for Call or 'put' for Put!")
            

    def gamma_calc(self, r, S, K, T, sigma):
        """Calculate delta of an option"""
        d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
        try:
            gamma_calc = stats.norm.pdf(d1, 0, 1)/(S*sigma*np.sqrt(T))
            return gamma_calc
        except:
            print("Please confirm option type, either 'call' for Call or 'put' for Put!")



if __name__ == "__main__":

    # Example usage
    S0 = 100     # Current stock price
    K = 105     # Strike price
    T = 1       # Time to maturity (1 year)
    r = 0.05    # Risk-free interest rate (5%)
    sigma = 0.2 # Volatility (20%)
    N = 1000
    M = 10000
    q = 0
    pricer = BlackScholesMCPricer(S0, K, r, T, sigma, option_type='call', N=N, M=M,q=q,)
    option_value, standard_error = pricer.monteCarlo()
    print(f"Call option value: ${option_value:.4f} ± {standard_error:.4f}")

    # optionInfo=Option_param(S,K,T)
    # processInfo=process_info(r,sigma,Diffusion_process(r,sigma,r).exp_RV)
    # bsmExact = BS_pricer(optionInfo,processInfo)
    # price = bsmExact.closed_formula()
    # print(price)
    # pricer.visualDist(price)
    





