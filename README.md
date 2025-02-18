## VolSnap-App

Download link: https://drive.google.com/file/d/1ZYx6ckULcbMd_iOdoq5PbTU9Rbittkr9/view?usp=sharing

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

- Features: Price surface, IV Surface, Greeks, Fourier Inversion and density plot

#### Variance Gamma Model:

- Methods: Fourier Inversion, FFT, PIDE, MC

- Features: Price surface, IV Surface, Greeks, Fourier Inversion and density plot


### WIP
- Current Issues: Some parameters will result in implied volatility surface plot not loading

- IR models and pricing IR derivatives (IR swaps, swaptions, caps/floors, etc.): Vasicek Model, CIR Model, Hull-White 

![Alt Text](https://github.com/Etrannn/VolSnap-App/blob/main/Images/IV_surface.png)
![Alt Text](https://github.com/Etrannn/VolSnap-App/blob/main/Images/paths.png)
![Alt Text](https://github.com/Etrannn/VolSnap-App/blob/main/Images/tree.png)
