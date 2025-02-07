#include<iostream>
#include<math.h>
#include<complex>
#include<algorithm>
#include<string>
#include <assert.h>
#include <vector>
using namespace std;
double trapezoidalMethod(vector<double> x, vector<double> y)
{
 int n = x.size();
 double answer = 0.0;
 for (int i = 1; i < n; i++)
  answer += 0.5 * (x[i] - x[i - 1]) * (y[i - 1] + y[i]);
 return answer;
}

double PI = 3.14159265358979323846264338327950288;
using namespace std;
//creates the function that will be integrated in the
//computation of the option price.
double Heston_integrand(complex<double> K, double X, double V0,
 double tau, double thet, double kappa, double SigmaV,
 double rho, double gamma)
{
 complex<double> thetaadj;
 double omega = kappa * thet;
 double ksi = SigmaV;
 double theta = kappa;
 complex<double> t((ksi * ksi) * tau / 2.0, 0.0);
 complex<double> a((2.0 * omega) / (ksi * ksi), 0.0);
 if (gamma == 1.0)
  thetaadj = complex<double>(theta, 0.0);
 else
  thetaadj = complex<double>((1.0 - gamma) * rho * ksi
   + sqrt(theta * theta - gamma * (1.0 - gamma) * ksi * ksi), 0.0);
 complex<double> im(0.0, 1.0);
 complex<double> re(1.0, 0.0);
 complex<double> b = (2.0 / (ksi * ksi)) * (thetaadj);
 complex<double> c = (K * K - im * K) / (ksi * ksi);
 complex<double> d = sqrt(b * b + 4.0 * c);
 complex<double> g = (b + d) / 2.0;
 complex<double> h = (b + d) / (b - d);
 complex<double> f1 = a * (t * g - log((1.0 - h * exp(t * d)) / (1.0 - h)));
 complex<double> f2 = g * ((1.0 - exp(t * d)) / (1.0 - h * exp(t * d)));
 complex<double> H = exp(f1 + f2 * V0);
 //function to be integrated
 complex<double> integrand = exp(-im * K * X) * (H / (K * K - im * K));
 return real(integrand);
}
 //setting up the numerical integration
 for (double phi = 0.000001; phi < kmax; phi += 0.2)
 {
  int_x[count] = phi;
  pass_phi = complex<double>(phi, ki);
  int_y[count] = Heston_integrand(pass_phi, X, V0,
   tau, thet, kappa, SigmaV, rho, gamma);
  count += 1;
 }
 //computing the price
 double callPrice = S * exp(-delta * tau) - (1.0 / (PI)) * K *
  exp(-r * tau) * trapezoidalMethod(int_x, int_y);
 double putPrice = callPrice + K * exp(-r * tau) -
  S * exp(-delta * tau);
 if (optType == "Call" || optType == "call")
  price = callPrice;
 else
  price = putPrice;
 return price;
}
//Greeks for the Heston model using finite differences
double HestonDelta(double S, double K, double r, double delta,
 double V0, double tau,
 double thet, double kappa, double SigmaV, double rho,
 double gamma, string optType)
{
 double alpha = 0.0001;
 return (HestonPrice(S + alpha, K, r, delta, V0, tau, thet, kappa,
  SigmaV, rho, gamma, optType) - HestonPrice(S, K, r, delta, V0,
   tau, thet, kappa, SigmaV, rho, gamma, optType)) / alpha;
}
double HestonGamma(double S, double K, double r, double delta,
 double V0, double tau,
 double thet, double kappa, double SigmaV, double rho,
 double gamma, string optType)
int main()
{
 double S = 90; //spot price
 double K = 90; //Strike price
 double r = 0.0425; //Risk-free interest rate
 double delta = 0; //Dividend yield
 double V0 = 0.01; //Initial volatility
 double tau = 0.5; //Time to maturity (years)
 double thet = 0.01; //Long-run volatility
 double kappa = 2; //Mean-reversion rate for volatility
 double SigmaV = 0.225; //Volatility of volatility
 double rho = 0; //Price-volatility correlation
 double gamma = 0; //Risk-aversion parameter
 string optType = "call"; //Option type (call or put)
 double price_, delta_, gamma_, rho_, vega, theta_;
 price_ = HestonPrice(S, K, r, delta, V0, tau, thet, kappa, SigmaV, rho, gamma, optType);
 delta_ = HestonDelta(S, K, r, delta, V0, tau, thet, kappa, SigmaV, rho, gamma, optType);
 gamma_ = HestonGamma(S, K, r, delta, V0, tau, thet, kappa, SigmaV, rho, gamma, optType);
 rho_ = HestonRho(S, K, r, delta, V0, tau, thet, kappa, SigmaV, rho, gamma, optType);
 vega = HestonVega(S, K, r, delta, V0, tau, thet, kappa, SigmaV, rho, gamma, optType);
 theta_ = HestonTheta(S, K, r, delta, V0, tau, thet, kappa, SigmaV, rho, gamma, optType);
 std::cout << price_ << endl;
 std::cout << delta_ << endl;
 std::cout << gamma_ << endl;
 std::cout << rho_ << endl;
 std::cout << vega << endl;
 std::cout << theta_ << endl;
 system("pause");
 return 0;
}