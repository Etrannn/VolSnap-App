from PyQt5.QtCore import Qt 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, 
    QWidget, QLabel, QLineEdit, QPushButton, 
    QHBoxLayout, QMessageBox, QFileDialog,
    QComboBox, QFormLayout, QStackedWidget,
    QCheckBox, QDockWidget
)

# from PyQt5.QtGui import QStandardItem, QStandardItemModel
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import sys

# Option models
import blackScholesPricer
import BatesPricer
import BinomialPricer
import HestonModelPricer
import MertonJumpDiffusionPricer
import Normal_InverseGaussianPricer
import VarianceGammaPricer

class VolSnap(QMainWindow):
    def __init__(self):
        super(VolSnap, self).__init__()
        self.setWindowTitle("VolSnap - Stochastic Model Visualiser")
        self.resize(1200, 800)

        # --- Central widget now contains only the chart ---
        central_widget = QWidget()
        central_layout = QVBoxLayout()
        self.figure = QLabel("---CHART WILL BE HERE SOON---")
        self.figure.setAlignment(Qt.AlignCenter)
        central_layout.addWidget(self.figure)
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

        # --- Create a dock widget for Option Inputs (left side) ---
        dock_inputs = QDockWidget("Model Inputs", self)
        dock_inputs.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock_inputs.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        dock_inputs.setStyleSheet("QDockWidget { background: grey; }")
        input_widget = QWidget()
        input_layout = QVBoxLayout()

        # Create model selector and add items
        self.model_selector = QComboBox()
        self.model_selector.addItems([
            "Black Scholes Merton", 
            "Bates(SVJ)", 
            "Binomial", 
            "Heston", 
            "Merton Jump Diffusion", 
            "Normal Inverse Gaussian", 
            "Variance Gamma"
        ])
        input_layout.addWidget(QLabel("Select Options Pricing Model:"))
        input_layout.addWidget(self.model_selector)

        self.bsm_widget = self.create_bsm_page()
        self.bates_widget = self.create_Bates_page()
        self.binomial_widget = self.create_binomial_page()
        self.heston_widget = self.create_heston_page()
        self.MJD_widget = self.create_MJD_page()
        self.NIG_widget = self.create_NIG_page()
        self.VG_widget = self.create_VG_page()

        self.stack = QStackedWidget()
        self.stack.addWidget(self.bsm_widget)
        self.stack.addWidget(self.bates_widget)
        self.stack.addWidget(self.binomial_widget)
        self.stack.addWidget(self.heston_widget)
        self.stack.addWidget(self.MJD_widget)
        self.stack.addWidget(self.NIG_widget)
        self.stack.addWidget(self.VG_widget)
        input_layout.addWidget(self.stack)

        self.model_selector.currentIndexChanged.connect(self.stack.setCurrentIndex)

        # Additional controls
        self.timer = QCheckBox("Enable Timer")
        input_layout.addWidget(self.timer)

        self.calc_button = QPushButton("Calculate")
        self.calc_button.clicked.connect(self.calculate_option_price)
        input_layout.addWidget(self.calc_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.reset_app)
        input_layout.addWidget(self.clear_button)

        input_widget.setLayout(input_layout)
        dock_inputs.setWidget(input_widget)

        # Add the inputs dock widget to the left dock area
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_inputs)

        # --- Create a dock widget for Prices and Greeks ---
        self.dock_prices_greeks = QDockWidget("Price and Greeks", self)
        self.dock_prices_greeks.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        # Allow the dock widget to be placed at the top or bottom
        self.dock_prices_greeks.setAllowedAreas(Qt.BottomDockWidgetArea)
        prices_greeks_widget = QWidget()
        prices_greeks_layout = QHBoxLayout()
        self.prices = QLabel("---PRICE INFO---")
        self.greeks = QLabel("---GREEKS INFO---")
        prices_greeks_layout.addWidget(self.prices, alignment=Qt.AlignCenter)
        prices_greeks_layout.addWidget(self.greeks, alignment=Qt.AlignCenter)
        prices_greeks_widget.setLayout(prices_greeks_layout)
        self.dock_prices_greeks.setWidget(prices_greeks_widget)
        self.dock_prices_greeks.setStyleSheet("font-size: 12px; font-weight: bold;")
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_prices_greeks)

        # --- Create a new dock widget for Visual Navigation (top right) ---
        self.dock_navigation = QDockWidget("", self)
        self.dock_navigation.setFeatures(QDockWidget.NoDockWidgetFeatures)
        # self.dock_navigation.setFixedHeight(60)
        # Allow this dock widget only at the top
        self.dock_navigation.setAllowedAreas(Qt.TopDockWidgetArea)
        nav_widget = QWidget()
        nav_layout = QHBoxLayout()
        nav_layout.setAlignment(Qt.AlignRight)  # Align buttons to the right
        self.btn_left = QPushButton("◄")
        self.btn_right = QPushButton("►")
        # Connect the arrow buttons to stub navigation functions
        self.btn_left.clicked.connect(self.navigate_left)
        self.btn_right.clicked.connect(self.navigate_right)
        nav_layout.addWidget(self.btn_left)
        nav_layout.addWidget(self.btn_right)
        nav_widget.setLayout(nav_layout)
        self.dock_navigation.setWidget(nav_widget)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_navigation)

    def navigate_left(self):
        # Stub function: implement functionality to scroll left through visuals
        print("Navigating left")

    def navigate_right(self):
        # Stub function: implement functionality to scroll right through visuals
        print("Navigating right")

    def create_bsm_page(self):
        widget = QWidget()
        self.bsm_inputs = {}
        layout = QFormLayout()
        self.bsm_inputs["S0"] = QLineEdit()
        layout.addRow("Underlying Price (S0):", self.bsm_inputs["S0"])
        self.bsm_inputs["K"] = QLineEdit()
        layout.addRow("Strike Price (K):", self.bsm_inputs["K"])
        self.bsm_inputs["sigma"] = QLineEdit()
        layout.addRow("Volatility (σ):", self.bsm_inputs["sigma"])
        self.bsm_inputs["r"] = QLineEdit()
        layout.addRow("Risk-Free Rate (r):", self.bsm_inputs["r"])
        self.bsm_inputs["T"] = QLineEdit()
        layout.addRow("Time to Maturity (T):", self.bsm_inputs["T"])
        self.bsm_inputs["payoff"] = QComboBox()
        self.bsm_inputs["payoff"].addItems(["call", "put"])
        layout.addRow("Option Type:", self.bsm_inputs["payoff"])
        widget.setLayout(layout)
        return widget

    def create_Bates_page(self):
        widget = QWidget()
        self.bates_inputs = {}
        layout = QFormLayout()
        self.bates_inputs["S0"] = QLineEdit()
        layout.addRow("Underlying Price (S0):", self.bates_inputs["S0"])
        self.bates_inputs["K"] = QLineEdit()
        layout.addRow("Strike Price (K):", self.bates_inputs["K"])
        self.bates_inputs["v0"] = QLineEdit()
        layout.addRow("Initial Variance (v0):", self.bates_inputs["v0"])
        self.bates_inputs["sigma"] = QLineEdit()
        layout.addRow("Volatility of Variance (σ):", self.bates_inputs["sigma"])
        self.bates_inputs["r"] = QLineEdit()
        layout.addRow("Risk-Free Rate (r):", self.bates_inputs["r"])
        self.bates_inputs["T"] = QLineEdit()
        layout.addRow("Time to Maturity (T):", self.bates_inputs["T"])
        self.bates_inputs["theta"] = QLineEdit()
        layout.addRow("Long-Run Variance (θ):", self.bates_inputs["theta"])
        self.bates_inputs["kappa"] = QLineEdit()
        layout.addRow("Mean Reversion Speed (k):", self.bates_inputs["kappa"])
        self.bates_inputs["rho"] = QLineEdit()
        layout.addRow("Correl. of Asset and Variance (ρ):", self.bates_inputs["rho"])
        self.bates_inputs["lam"] = QLineEdit()
        layout.addRow("Jump Intensity (λ):", self.bates_inputs["lam"])
        self.bates_inputs["muJ"] = QLineEdit()
        layout.addRow("Mean Jump Size (μ):", self.bates_inputs["muJ"])
        self.bates_inputs["delta"] = QLineEdit()
        layout.addRow("Volatility of Jump (δ):", self.bates_inputs["delta"])
        self.bates_inputs["payoff"] = QComboBox()
        self.bates_inputs["payoff"].addItems(["call", "put"])
        layout.addRow("Option Type:", self.bates_inputs["payoff"])
        widget.setLayout(layout)
        return widget

    def create_binomial_page(self):
        widget = QWidget()
        self.binomial_inputs = {}
        layout = QFormLayout()
        self.binomial_inputs["S0"] = QLineEdit()
        layout.addRow("Underlying Price (S0):", self.binomial_inputs["S0"])
        self.binomial_inputs["K"] = QLineEdit()
        layout.addRow("Strike Price (K):", self.binomial_inputs["K"])
        self.binomial_inputs["sigma"] = QLineEdit()
        layout.addRow("Volatility (σ):", self.binomial_inputs["sigma"])
        self.binomial_inputs["r"] = QLineEdit()
        layout.addRow("Risk-Free Rate (r):", self.binomial_inputs["r"])
        self.binomial_inputs["T"] = QLineEdit()
        layout.addRow("Time to Maturity (T):", self.binomial_inputs["T"])
        self.binomial_inputs["Steps"] = QLineEdit()
        layout.addRow("Steps:", self.binomial_inputs["Steps"])
        self.binomial_inputs["payoff"] = QComboBox()
        self.binomial_inputs["payoff"].addItems(["call", "put"])
        layout.addRow("Option Type:", self.binomial_inputs["payoff"])
        self.binomial_inputs["style"] = QComboBox()
        self.binomial_inputs["style"].addItems(["European", "American"])
        layout.addRow("Option Style:", self.binomial_inputs["style"])
        widget.setLayout(layout)
        return widget

    def create_heston_page(self):
        widget = QWidget()
        self.heston_inputs = {}
        layout = QFormLayout()
        self.heston_inputs["S0"] = QLineEdit()
        layout.addRow("Underlying Price (S0):", self.heston_inputs["S0"])
        self.heston_inputs["K"] = QLineEdit()
        layout.addRow("Strike Price (K):", self.heston_inputs["K"])
        self.heston_inputs["v0"] = QLineEdit()
        layout.addRow("Initial Variance (v0):", self.heston_inputs["v0"])
        self.heston_inputs["sigma"] = QLineEdit()
        layout.addRow("Volatility of Variance (σ):", self.heston_inputs["sigma"])
        self.heston_inputs["r"] = QLineEdit()
        layout.addRow("Risk-Free Rate (r):", self.heston_inputs["r"])
        self.heston_inputs["T"] = QLineEdit()
        layout.addRow("Time to Maturity (T):", self.heston_inputs["T"])
        self.heston_inputs["theta"] = QLineEdit()
        layout.addRow("Long-Run Variance (θ):", self.heston_inputs["theta"])
        self.heston_inputs["kappa"] = QLineEdit()
        layout.addRow("Mean Reversion Speed (k):", self.heston_inputs["kappa"])
        self.heston_inputs["rho"] = QLineEdit()
        layout.addRow("Correl. of Asset and Variance (ρ):", self.heston_inputs["rho"])
        self.heston_inputs["payoff"] = QComboBox()
        self.heston_inputs["payoff"].addItems(["call", "put"])
        layout.addRow("Option Type:", self.heston_inputs["payoff"])
        widget.setLayout(layout)
        return widget

    def create_MJD_page(self):
        widget = QWidget()
        self.MJD_inputs = {}
        layout = QFormLayout()
        self.MJD_inputs["S0"] = QLineEdit()
        layout.addRow("Underlying Price (S0):", self.MJD_inputs["S0"])
        self.MJD_inputs["K"] = QLineEdit()
        layout.addRow("Strike Price (K):", self.MJD_inputs["K"])
        self.MJD_inputs["sigma"] = QLineEdit()
        layout.addRow("Volatility (σ):", self.MJD_inputs["sigma"])
        self.MJD_inputs["r"] = QLineEdit()
        layout.addRow("Risk-Free Rate (r):", self.MJD_inputs["r"])
        self.MJD_inputs["T"] = QLineEdit()
        layout.addRow("Time to Maturity (T):", self.MJD_inputs["T"])
        self.MJD_inputs["lam"] = QLineEdit()
        layout.addRow("Jump Intensity (λ):", self.MJD_inputs["lam"])
        self.MJD_inputs["muJ"] = QLineEdit()
        layout.addRow("Mean Jump Size (μ):", self.MJD_inputs["muJ"])
        self.MJD_inputs["sigJ"] = QLineEdit()
        layout.addRow("Volatility of Jump (δ):", self.MJD_inputs["sigJ"])
        self.MJD_inputs["payoff"] = QComboBox()
        self.MJD_inputs["payoff"].addItems(["call", "put"])
        layout.addRow("Option Type:", self.MJD_inputs["payoff"])
        widget.setLayout(layout)
        return widget

    def create_NIG_page(self):
        widget = QWidget()
        self.NIG_inputs = {}
        layout = QFormLayout()
        self.NIG_inputs["S0"] = QLineEdit()
        layout.addRow("Underlying Price (S0):", self.NIG_inputs["S0"])
        self.NIG_inputs["K"] = QLineEdit()
        layout.addRow("Strike Price (K):", self.NIG_inputs["K"])
        self.NIG_inputs["sigma"] = QLineEdit()
        layout.addRow("Volatility of Variance (σ):", self.NIG_inputs["sigma"])
        self.NIG_inputs["r"] = QLineEdit()
        layout.addRow("Risk-Free Rate (r):", self.NIG_inputs["r"])
        self.NIG_inputs["T"] = QLineEdit()
        layout.addRow("Time to Maturity (T):", self.NIG_inputs["T"])
        self.NIG_inputs["theta"] = QLineEdit()
        layout.addRow("Drift of the Brownian motion (θ):", self.NIG_inputs["theta"])
        self.NIG_inputs["kappa"] = QLineEdit()
        layout.addRow("Variance of the Gamma process (k):", self.NIG_inputs["kappa"])
        self.NIG_inputs["payoff"] = QComboBox()
        self.NIG_inputs["payoff"].addItems(["call", "put"])
        layout.addRow("Option Type:", self.NIG_inputs["payoff"])
        widget.setLayout(layout)
        return widget

    def create_VG_page(self):
        widget = QWidget()
        self.VG_inputs = {}
        layout = QFormLayout()
        self.VG_inputs["S0"] = QLineEdit()
        layout.addRow("Underlying Price (S0):", self.VG_inputs["S0"])
        self.VG_inputs["K"] = QLineEdit()
        layout.addRow("Strike Price (K):", self.VG_inputs["K"])
        self.VG_inputs["sigma"] = QLineEdit()
        layout.addRow("Volatility of Variance (σ):", self.VG_inputs["sigma"])
        self.VG_inputs["r"] = QLineEdit()
        layout.addRow("Risk-Free Rate (r):", self.VG_inputs["r"])
        self.VG_inputs["T"] = QLineEdit()
        layout.addRow("Time to Maturity (T):", self.VG_inputs["T"])
        self.VG_inputs["theta"] = QLineEdit()
        layout.addRow("Drift of the Brownian motion (θ):", self.VG_inputs["theta"])
        self.VG_inputs["kappa"] = QLineEdit()
        layout.addRow("Variance of the Gamma process (k):", self.VG_inputs["kappa"])
        self.VG_inputs["payoff"] = QComboBox()
        self.VG_inputs["payoff"].addItems(["call", "put"])
        layout.addRow("Option Type:", self.VG_inputs["payoff"])
        widget.setLayout(layout)
        return widget

    def calculate_option_price(self):
        model_index = self.model_selector.currentIndex()
        try:
            if model_index == 0:  # BSM Model
                S0 = float(self.bsm_inputs["S0"].text())
                K = float(self.bsm_inputs["K"].text())
                sigma = float(self.bsm_inputs["sigma"].text())
                r = float(self.bsm_inputs["r"].text())
                T = float(self.bsm_inputs["T"].text())
                payoff = self.bsm_inputs["payoff"].currentText()

                option_info = blackScholesPricer.Option_param(S0=S0, K=K, T=T, payoff=payoff)
                process_info_obj = blackScholesPricer.Diffusion_process(r=r, sig=sigma)

                pricer = blackScholesPricer.BS_pricer(option_info, process_info_obj)

                Closedprice = pricer.closed_formula()
                FIprice = pricer.Fourier_inversion()
                FFTprice = pricer.FFT(K)
                pdePrice, pdeTime = pricer.PDE_price(steps=(5000, 4000), Time=True, solver="splu")
                LSMPrice = pricer.LSM(N=500, paths=5000)
                greek = pricer.calculate_greeks()

                self.prices.setText(
                    f"Closed-Form {payoff.capitalize()} Price: {Closedprice:.4f}\n"
                    f"Fourier Inversion {payoff.capitalize()} Price: {FIprice:.4f}\n"
                    f"FFT {payoff.capitalize()} Price: {FFTprice:.4f}\n"
                    f"PDE {payoff.capitalize()} Price: {pdePrice:.4f}\n"
                    f"Longstaff-Schwartz {payoff.capitalize()} Price (American): {LSMPrice:.4f}"
                )
                self.greeks.setText(f"{greek}")

                

            elif model_index == 1:  # Bates Model
                S0 = float(self.bates_inputs["S0"].text())
                K = float(self.bates_inputs["K"].text())
                sigma = float(self.bates_inputs["sigma"].text())
                r = float(self.bates_inputs["r"].text())
                T = float(self.bates_inputs["T"].text())
                theta = float(self.bates_inputs["theta"].text())
                kappa = float(self.bates_inputs["kappa"].text())
                rho = float(self.bates_inputs["rho"].text())
                lam = float(self.bates_inputs["lam"].text())
                muJ = float(self.bates_inputs["muJ"].text())
                delta = float(self.bates_inputs["delta"].text())
                v0 = float(self.bates_inputs["v0"].text())
                payoff = self.bates_inputs["payoff"].currentText()

                option_info = BatesPricer.Option_param(S0, K, T, v0, payoff=payoff)
                Bates_process = BatesPricer.process_info(r, sigma, theta, kappa, rho, lam, muJ, delta)

                pricer = BatesPricer.Bates_pricer(option_info, Bates_process)
                if not pricer.check_feller_condition(kappa, theta, sigma):
                    QMessageBox.critical(self, "Error", "The Feller condition is NOT satisfied.")
                FIprice = pricer.Fourier_inversion()
                FFTprice = pricer.FFT(K)
                greeks = pricer.calculate_greeks()

                self.prices.setText(
                    f"Fourier Inversion {payoff.capitalize()} Price: {FIprice:.4f}\n"
                    f"FFT {payoff.capitalize()} Price: {FFTprice:.4f}"
                )
                self.greeks.setText(f"{greeks}")
            elif model_index == 2:  # Binomial Model
                S0 = float(self.binomial_inputs["S0"].text())
                K = float(self.binomial_inputs["K"].text())
                sigma = float(self.binomial_inputs["sigma"].text())
                r = float(self.binomial_inputs["r"].text())
                T = float(self.binomial_inputs["T"].text())
                steps = int(self.binomial_inputs["Steps"].text())
                payoff = self.binomial_inputs["payoff"].currentText()
                style = self.binomial_inputs["style"].currentText()
                pricer = BinomialPricer.BinomialPricer(S0, K, T, r, sigma, steps, payoff, style)

                treePrice = pricer.price()
                greeks = pricer.calculate_greeks()

                self.prices.setText(
                    f"Tree price {payoff.capitalize()} Price: {treePrice:.4f}"
                )
                self.greeks.setText(f"{greeks}")
            elif model_index == 3:  # Heston Model
                S0 = float(self.heston_inputs["S0"].text())
                K = float(self.heston_inputs["K"].text())
                v0 = float(self.heston_inputs["v0"].text())
                sigma = float(self.heston_inputs["sigma"].text())
                r = float(self.heston_inputs["r"].text())
                T = float(self.heston_inputs["T"].text())
                theta = float(self.heston_inputs["theta"].text())
                kappa = float(self.heston_inputs["kappa"].text())
                rho = float(self.heston_inputs["rho"].text())
                payoff = self.heston_inputs["payoff"].currentText()

                optionInfo = HestonModelPricer.Option_param(S0, K, T, v0, payoff=payoff)
                HestonProcess = HestonModelPricer.process_info(r, sigma, theta, kappa, rho)
                pricer = HestonModelPricer.Heston_pricer(optionInfo, HestonProcess)
                if not pricer.check_feller_condition(kappa, theta, sigma):
                    QMessageBox.critical(self, "Error", "The Feller condition is NOT satisfied.")
                FIprice = pricer.Fourier_inversion()
                FFTprice = pricer.FFT(K)
                greeks = pricer.calculate_greeks()
                self.prices.setText(
                    f"Fourier Inversion {payoff.capitalize()} Price: {FIprice:.4f}\n"
                    f"FFT {payoff.capitalize()} Price: {FFTprice:.4f}"
                )
                self.greeks.setText(f"{greeks}")
            elif model_index == 4:  # MJD Model
                S0 = float(self.MJD_inputs["S0"].text())
                K = float(self.MJD_inputs["K"].text())
                sigma = float(self.MJD_inputs["sigma"].text())
                r = float(self.MJD_inputs["r"].text())
                T = float(self.MJD_inputs["T"].text())
                lam = float(self.MJD_inputs["lam"].text())
                muJ = float(self.MJD_inputs["muJ"].text())
                sigJ = float(self.MJD_inputs["sigJ"].text())
                payoff = self.MJD_inputs["payoff"].currentText()

                optionInfo = MertonJumpDiffusionPricer.Option_param(S0, K, T, payoff=payoff)
                processInfo = MertonJumpDiffusionPricer.process_info(
                    r,
                    sigma,
                    lam,
                    muJ,
                    sigJ,
                    MertonJumpDiffusionPricer.Merton_process(r, sigma, lam, muJ, sigJ).exp_RV
                )
                pricer = MertonJumpDiffusionPricer.Merton_pricer(optionInfo, processInfo)

                closedform = pricer.closed_formula()
                FIprice = pricer.Fourier_inversion()
                FFTprice = pricer.FFT(K)
                pidePrice, pideTime = pricer.PIDE_price(steps=(5000, 4000), Time=True)
                greek = pricer.calculate_greeks()

                self.prices.setText(
                    f"Closed-Form {payoff.capitalize()} Price: {closedform:.4f}\n"
                    f"Fourier Inversion {payoff.capitalize()} Price: {FIprice:.4f}\n"
                    f"FFT {payoff.capitalize()} Price: {FFTprice:.4f}\n"
                    f"PIDE {payoff.capitalize()} Price: {pidePrice:.4f}"
                )
                self.greeks.setText(f"{greek}")
            elif model_index == 5:  # NIG Model
                S0 = float(self.NIG_inputs["S0"].text())
                K = float(self.NIG_inputs["K"].text())
                sigma = float(self.NIG_inputs["sigma"].text())
                r = float(self.NIG_inputs["r"].text())
                T = float(self.NIG_inputs["T"].text())
                theta = float(self.NIG_inputs["theta"].text())
                kappa = float(self.NIG_inputs["kappa"].text())
                payoff = self.NIG_inputs["payoff"].currentText()

                optionInfo = Normal_InverseGaussianPricer.Option_param(S0, K, T, payoff=payoff)
                processInfo = Normal_InverseGaussianPricer.process_info(
                    r,
                    sigma,
                    theta,
                    kappa,
                    Normal_InverseGaussianPricer.NIG_process(r, sigma, theta, kappa).exp_RV
                )
                pricer = Normal_InverseGaussianPricer.NIG_pricer(optionInfo, processInfo)
                FIprice = pricer.Fourier_inversion()
                FFTprice = pricer.FFT(K)
                pidePrice, pideTime = pricer.PIDE_price((5000, 4000), Time=True)
                greek = pricer.calculate_greeks()
                self.prices.setText(
                    f"Fourier Inversion {payoff.capitalize()} Price: {FIprice:.4f}\n"
                    f"FFT {payoff.capitalize()} Price: {FFTprice:.4f}\n"
                    f"PIDE {payoff.capitalize()} Price: {pidePrice:.4f}"
                )
                self.greeks.setText(f"{greek}")
            elif model_index == 6:  # VG Model
                S0 = float(self.VG_inputs["S0"].text())
                K = float(self.VG_inputs["K"].text())
                sigma = float(self.VG_inputs["sigma"].text())
                r = float(self.VG_inputs["r"].text())
                T = float(self.VG_inputs["T"].text())
                theta = float(self.VG_inputs["theta"].text())
                kappa = float(self.VG_inputs["kappa"].text())
                payoff = self.VG_inputs["payoff"].currentText()

                optionInfo = VarianceGammaPricer.Option_param(S0, K, T, payoff=payoff)
                processInfo = VarianceGammaPricer.process_info(
                    r,
                    sigma,
                    theta,
                    kappa,
                    VarianceGammaPricer.VG_process(r, sigma, theta, kappa).exp_RV
                )
                pricer = VarianceGammaPricer.VG_pricer(optionInfo, processInfo)
                FIprice = pricer.Fourier_inversion()
                FFTprice = pricer.FFT(K)
                pidePrice, pideTime = pricer.PIDE_price((5000, 4000), Time=True)
                greek = pricer.calculate_greeks()
                self.prices.setText(
                    f"Fourier Inversion {payoff.capitalize()} Price: {FIprice:.4f}\n"
                    f"FFT {payoff.capitalize()} Price: {FFTprice:.4f}\n"
                    f"PIDE {payoff.capitalize()} Price: {pidePrice:.4f}"
                )
                self.greeks.setText(f"{greek}")
            else:
                self.figure.setText("Pricing not implemented for the selected model yet.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")


    def reset_app(self):
        self.model_selector.setCurrentIndex(0)

        input_dicts = [
            self.bsm_inputs,
            self.bates_inputs,
            self.binomial_inputs,
            self.heston_inputs,
            self.MJD_inputs,
            self.NIG_inputs,
            self.VG_inputs,
        ]

        for inp_dict in input_dicts:
            for widget in inp_dict.values():
                if isinstance(widget, QLineEdit):
                    widget.clear()
                elif isinstance(widget, QComboBox):
                    widget.setCurrentIndex(0)

        self.figure.setText("---CHART WILL BE HERE SOON---")
        self.prices.setText("---PRICE INFO---")
        self.greeks.setText("---GREEKS INFO---")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VolSnap()
    window.show()
    sys.exit(app.exec_())









