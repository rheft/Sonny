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
from sklearn.tree import DecisionTreeRegressor

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
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Visualize to results
x_grid = np.arange(min(X), max(X), 0.01)
x_grid = x_grid.reshape((len(x_grid)), 1)
plt.scatter(X, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('Linear Regression vs. Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predict new employee
regressor.predict(np.array(6.5).reshape(1, -1))
