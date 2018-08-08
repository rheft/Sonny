#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 07:41:34 2018

@author: robheft
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Compare linear regressor to polynomial regressor
# Linear Model
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(X, y)

# Polynomial Model
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree = 4)
X_poly = polyReg.fit_transform(X)

linReg2 = LinearRegression()
linReg2.fit(X_poly, y)

# Visualize the Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X, linReg.predict(X), color='blue')
plt.title('Truth or Bluff Linear')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show

# Predict new Result with Linear Regression
linReg.predict(6.5)

# Visualize the Polynomial regression results
xGrid = np.arange(min(X), max(X), 0.1)
xGrid = xGrid.reshape((len(xGrid), 1))
plt.scatter(X, y, color='red')
plt.plot(xGrid, linReg2.predict(polyReg.fit_transform(xGrid)), color='blue')
plt.title('Truth or Bluff Polynomial')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show

# Predict new Result with Polynomial regression
linReg2.predict(polyReg.fit_transform(6.5))