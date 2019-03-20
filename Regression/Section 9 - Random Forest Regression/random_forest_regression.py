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
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

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

# Decision Tree Regression
regressor_10 = RandomForestRegressor(n_estimators = 10, random_state=0)
regressor_10.fit(X, y)
regressor_50 = RandomForestRegressor(n_estimators = 50, random_state=0)
regressor_50.fit(X, y)
regressor_100 = RandomForestRegressor(n_estimators = 100, random_state=0)
regressor_100.fit(X, y)
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Visualize to results
x_grid = np.arange(min(X), max(X), 0.01)
x_grid = x_grid.reshape((len(x_grid)), 1)
plt.scatter(X, y, color='red')
plt.plot(x_grid, regressor_10.predict(x_grid), color='blue')
plt.plot(x_grid, regressor_50.predict(x_grid), color='green')
plt.plot(x_grid, regressor_100.predict(x_grid), color='purple')
plt.plot(x_grid, lin_reg.predict(x_grid), color='black')
plt.title('Linear Regression vs. Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predict new employee
regressor_10.predict(np.array(6.5).reshape(1, -1))
regressor_50.predict(np.array(6.5).reshape(1, -1))
regressor_100.predict(np.array(6.5).reshape(1, -1))
