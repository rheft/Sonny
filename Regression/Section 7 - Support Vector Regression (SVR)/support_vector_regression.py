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
from sklearn.preprocessing import StandardScaler

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
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# Polynomial Regression
polynomial_regressor = PolynomialFeatures(degree=4)
X_poly = polynomial_regressor.fit_transform(X)
multiple_linear_regressor = LinearRegression()
multiple_linear_regressor.fit(X_poly, y)

# Compare predictions
linear_regressor.predict(np.array(6.5).reshape(1, -1))
multiple_linear_regressor.predict(polynomial_regressor.fit_transform(np.array(6.5).reshape(1, -1)))

# Need to do feature scaling for svr
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))

# Fit SVR
svr_regressor = SVR(kernel='rbf')
svr_regressor.fit(X, y)
y_pred = sc_y.inverse_transform(svr_regressor.predict(sc_X.transform(np.array([[6.5]]))))
