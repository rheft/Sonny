import sys
import os
from dotenv import load_dotenv, find_dotenv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Import lib files
envs = load_dotenv(find_dotenv())
file = os.getenv("lib")
sys.path.insert(0, file)
from utils import LoadData
from preprocessing import PreProcessing

# Load data
dataset = LoadData("Position_Salaries.csv").data

# Split the dataset
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Linear regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
# Polynomial Regression
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
multi_reg = LinearRegression()
multi_reg.fit(X_poly, y)

# Visualize to results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.plot(X, multi_reg.predict(poly_reg.fit_transform(X)), color='green')
plt.title('Linear Regression vs. Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predict new employee
lin_reg.predict(np.array(6.5).reshape(1, -1))
multi_reg.predict(poly_reg.fit_transform(np.array(6.5).reshape(1, -1)))
