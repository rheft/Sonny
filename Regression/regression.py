# Imports
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Import lib files
envs = load_dotenv(find_dotenv())
file = os.getenv("lib")
sys.path.insert(0, file)
from utils import LoadData
from preprocessing import PreProcessing

class Regression():
    def __init__(self, features, target):
        self.features = features
        self.targets = targets
        self.processor = PreProcessing()

    def simple_linear_regression(self):
        linear_regression = LinearRegression()
        linear_regression.fit(self.features, self.targets)
        return linear_regression

    def polynomial_regression(self, polynomial=4):
        polynomializer = PolynomialFeatures(degree=polynomial)
        X_poly = polynomializer.fit_transform(self.features)
        poly_regression = LinearRegression()
        poly_regression.fit(X_poly, self.targets)
        return poly_regression

    def support_vector_regression(self, kernel='rbf'):
        X = self.processor.fit_scaler(self.features)
        y = self.processor.fit_scaler(self.targets)
        svr_regressor = SVR(kernel=kernel)
        svr_regressor.fit(X, y)
        return svr_regressor

    def decision_tree_regression(self):
        decision_tree_regressor = DecisionTreeRegressor()
        decision_tree_regressor.fit(self.features, self.targets)
        return decision_tree_regressor

    def random_forest_regression(self, estimators = 100):
        random_forest_regression = RandomForestRegressor(n_estimators=estimators)
        random_forest_regression.fit(self.features, self.targets)
        return random_forest_regression


if __name__ == '__main__':

    # Load data
    dataset = LoadData("Position_Salaries.csv").data

    # Split the dataset
    X = dataset.iloc[:, 1:2].values
    y = dataset.iloc[:, 2].values

    # create regressor object
    regressor = Regression(features=X, targets=y)
