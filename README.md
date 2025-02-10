## VolSnap-App

An application that outputs various pricing methods, greeks and plots of various options pricing models.
Credit to cantaro86 for BSM, Heston, Merton, VG and NG pricing models

### Current models:

#### BSM model: 

- Methods: Analytical closed solution, Fourier Inversion, FFT, PDE, Longstaff-Schwartz, MC 

- Features: Price surface, IV Surface, Option price & P/L Heat map, Greeks, Stock Path simulations

#### Bates (SVJ) Model:

- Methods:  Fourier Inversion, FFT, MC

- Features:  Price surface, IV Surface, Option price & P/L Heat map, Greeks, Stock Path simulations

#### Binomial (CRR) Model:

- Methods: Tree method

- Features: Stock Price Tree, Option Price Tree, Volatility Impact on Option Price 

#### Heston Model:

- Methods:  Fourier Inversion, FFT, MC

- Features: Price surface, IV Surface, Option price & P/L Heat map, Greeks, Stock Path simulations


#### Merton Jump Diffusion: 

- Methods: Analytical closed solution, Fourier Inversion, FFT, PIDE, MC 

- Features: Price surface, IV Surface, Option price & P/L Heat map, Greeks, Stock Path simulations

#### Normal Inverse Gaussian Model:

- Methods: Fourier Inversion, FFT, PIDE, MC

- Features: Price surface, IV Surface, Greeks, Stock Path simulations

#### Variance Gamma Model:

- Methods: Fourier Inversion, FFT, PIDE, MC

- Features: Price surface, IV Surface, Greeks, Stock Path simulations


### WIP
- Current Issues: Some parameters will result in implied volatility surface plot not loading

- IR models and pricing IR derivatives (IR swaps, swaptions, caps/floors, etc.): Vasicek Model, CIR Model, Hull-White 

<img src="C:\Users\61430\Pictures\Screenshots\Screenshot 2025-02-11 081206.png" alt="My Image" width="400">
<img src="C:\Users\61430\Pictures\Screenshots\Screenshot 2025-02-11 081306.png" alt="My Image" width="400">
<img src="C:\Users\61430\Pictures\Screenshots\Screenshot 2025-02-11 081548.png" alt="My Image" width="400">



