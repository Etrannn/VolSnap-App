# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

# # https://www.linkedin.com/pulse/sabr-volatility-model-unlocking-realistic-market-behavior-yadav-r3kkc/




# mktPrices = pd.read_csv("https://raw.githubusercontent.com/codearmo/data/master/calls_calib_example.csv")
# print(mktPrices['Bid'].head(10))


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import CubicSpline

# class HullWhiteMC:
#     def __init__(self, kappa, sigma, r0, T, dt, n_paths, maturities, yields):
#         self.kappa = kappa
#         self.sigma = sigma
#         self.r0 = r0
#         self.T = T
#         self.dt = dt
#         self.n_paths = n_paths
#         self.n_steps = int(T / dt)

#         # Build a cubic spline interpolator for f(0, t)
#         self.forward_curve = self.build_forward_curve(maturities, yields)

#     def build_forward_curve(self, maturities, yields):
#         """ Compute the instantaneous forward rate curve f(0,t). """
#         # Convert yields to instantaneous forward rates
#         f_t = np.gradient(yields, maturities) * maturities + yields
#         return CubicSpline(maturities, f_t, bc_type="clamped")

#     def theta(self, t):
#         """ Compute term-structure based theta(t). """
#         f_t = self.forward_curve(t)
#         df_dt = self.forward_curve.derivative()(t)
#         return df_dt + self.kappa * f_t + (self.sigma**2 / (2 * self.kappa)) * (1 - np.exp(-2 * self.kappa * t))

#     def simulate_paths(self):
#         """ Monte Carlo simulation of Hull-White short rate model with term-structure-based theta(t). """
#         times = np.linspace(0, self.T, self.n_steps)
#         rates = np.zeros((self.n_paths, self.n_steps))
#         rates[:, 0] = self.r0
        
#         for i in range(1, self.n_steps):
#             t = times[i-1]
#             dW = np.sqrt(self.dt) * np.random.randn(self.n_paths)  # Brownian motion increments
#             dr = self.kappa * (self.theta(t) - rates[:, i-1]) * self.dt + self.sigma * dW
#             rates[:, i] = rates[:, i-1] + dr
        
#         return times, rates

#     def plot_paths(self):
#         """ Plot simulated interest rate paths and their mean. """
#         times, rates = self.simulate_paths()
        
#         plt.figure(figsize=(10, 6))
#         for i in range(self.n_paths):
#             plt.plot(times, rates[i, :], lw=0.7, alpha=0.6, color="blue")
        
#         plt.plot(times, np.mean(rates, axis=0), lw=2, color="red", label="Mean Path")
#         plt.xlabel("Time (Years)")
#         plt.ylabel("Short Rate (r)")
#         plt.title("Hull-White Short Rate Simulation (MC) with Term Structure-based θ(t)")
#         plt.legend()
#         plt.grid()
#         plt.show()

# # Example term structure (maturities in years, yields in %)
# maturities = np.array([0.5, 1, 2, 5, 10, 20])
# yields = np.array([0.02, 0.025, 0.03, 0.035, 0.04, 0.045])  # Example yield curve

# # Model parameters
# kappa = 0.4    # Mean reversion speed
# sigma = 0.05   # Volatility
# r0 = 0.02       # Initial short rate
# T = 10          # Maturity in years
# dt = 0.01       # Time step
# n_paths = 500   # Number of Monte Carlo paths

# # Run simulation
# hw_mc = HullWhiteMC(kappa, sigma, r0, T, dt, n_paths, maturities, yields)
# hw_mc.plot_paths()


from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QLineEdit, QPushButton, QTreeView, QHBoxLayout, QMessageBox, QFileDialog
from PyQt5.QtGui import QStandardItem, QStandardItemModel
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# class FinanceApp(QMainWindow):
#     def __init__(self):
#         super(FinanceApp, self).__init__()

#         # Set window title
#         self.setWindowTitle("Finance App")

#         # Set window size
#         self.setGeometry(100, 100, 800, 600)

#         # Create main widget
#         main_widget = QWidget()

#         # Create vertical layout for main widget
#         self.layout = QVBoxLayout()
#         row1 = QHBoxLayout()
#         row2 = QHBoxLayout()
#         # Create input fields and labels
#         self.interest_rate_label = QLabel("Interest Rate (%)")
#         self.interest_rate_input = QLineEdit()
#         row1.addWidget(self.interest_rate_label)
#         row1.addWidget(self.interest_rate_input)

#         self.initial_investment_label = QLabel("Initial Investment ($)")
#         self.initial_investment_input = QLineEdit()
#         row1.addWidget(self.initial_investment_label)
#         row1.addWidget(self.initial_investment_input)

#         self.num_years_label = QLabel("Number of Years")
#         self.num_years_input = QLineEdit()
#         row1.addWidget(self.num_years_label)
#         row1.addWidget(self.num_years_input)

        
#         # Create TreeView for results
#         self.model = QStandardItemModel()
#         self.tree_view = QTreeView()
#         self.tree_view.setModel(self.model)
#         row2.addWidget(self.tree_view)
        
#         # Create plot widget
#         self.figure = plt.figure()
#         self.canvas = FigureCanvas(self.figure)
#         row2.addWidget(self.canvas)

#         self.layout.addLayout(row1)
#         self.layout.addLayout(row2)
#         # Create calculate button
#         self.calculate_button = QPushButton("Calculate")
#         self.layout.addWidget(self.calculate_button)

#         # Create reset button
#         self.reset_button = QPushButton("Reset")
#         self.layout.addWidget(self.reset_button)
        
        
        
        

#         # Set main widget layout
#         main_widget.setLayout(self.layout)

#         # Set main widget as central widget
#         self.setCentralWidget(main_widget)

#         # Connect buttons to slots
#         self.calculate_button.clicked.connect(self.calculate_compound_interest)
#         self.reset_button.clicked.connect(self.reset_app)

#     def calculate_compound_interest(self):
#     # Get input values
#         interest_rate = float(self.interest_rate_input.text())
#         initial_investment = float(self.initial_investment_input.text())
#         num_years = int(self.num_years_input.text())

#         # Clear previous results
#         self.model.clear()

#         # Add column headers to the tree view
#         self.model.setHorizontalHeaderLabels(["Year", "Total"])

#         # Calculate compound interest and add results to tree view
#         total = initial_investment
#         for year in range(1, num_years + 1):
#             total += total * (interest_rate / 100)
#             item_year = QStandardItem(str(year))
#             item_total = QStandardItem("{:.2f}".format(total))
#             self.model.appendRow([item_year, item_total])

#         # Update chart with results
#         self.figure.clear()
#         ax = self.figure.add_subplot(111)
#         years = list(range(1, num_years + 1))
#         totals = [initial_investment * (1 + interest_rate / 100) ** year for year in years]     # Fix the formula here
#         ax.plot(years, totals)
#         ax.set_xlabel("Year")
#         ax.set_ylabel("Total")
#         self.canvas.draw()
        
#         self.save_button = QPushButton("Save")
#         self.save_button.clicked.connect(self.save_results)
#         self.layout.addWidget(self.save_button)
        

#     def save_results(self):
#     # Open a file dialog to choose a directory
#         dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
#         if dir_path:
#             # Create a subfolder within the selected directory
#             folder_path = os.path.join(dir_path, "Results")
#             os.makedirs(folder_path, exist_ok=True)

#             # Save the results to a CSV file within the subfolder
#             file_path = os.path.join(folder_path, "results.csv")
#             with open(file_path, "w") as file:
#                 file.write("Year,Total\n")
#                 for row in range(self.model.rowCount()):
#                     year = self.model.index(row, 0).data()
#                     total = self.model.index(row, 1).data()
#                     file.write("{},{}\n".format(year, total))

#             # Show a message box to indicate successful save
#             QMessageBox.information(self, "Save Results", "Results saved successfully in '{}'".format(folder_path))
#         else:
#             QMessageBox.warning(self, "Save Results", "No directory selected.")


#     def reset_app(self):
#         # Clear input fields and tree view
#         self.interest_rate_input.clear()
#         self.initial_investment_input.clear()
#         self.num_years_input.clear()
#         self.model.clear()
#         self.figure.clear()
#         self.canvas.draw()

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     finance_app = FinanceApp()
#     finance_app.show()
#     app.exec_()

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QCheckBox, QLabel

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Checkbox Example")
        self.setGeometry(100, 100, 300, 200)

        # Create layout
        layout = QVBoxLayout()

        # Create a checkbox
        self.checkbox = QCheckBox("Enable Feature")
        self.checkbox.stateChanged.connect(self.checkbox_toggled)

        # Create a label to display status
        self.label = QLabel("Feature is OFF")

        # Add widgets to layout
        layout.addWidget(self.checkbox)
        layout.addWidget(self.label)

        self.setLayout(layout)

    def checkbox_toggled(self, state):
        if state == 2:  # Checked (Qt.Checked)
            self.label.setText("Feature is ON")
        else:  # Unchecked (Qt.Unchecked)
            self.label.setText("Feature is OFF")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())
