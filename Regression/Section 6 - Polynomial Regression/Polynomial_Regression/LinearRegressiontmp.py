#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 07:55:23 2018

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

'''
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

'''

# Polynomial Model
# Create your regressor here

# Predict new Result with Polynomial regression
y_pred = regressor.predict(6.5)

# Visualize the regression results
xGrid = np.arange(min(X), max(X), 0.1)
xGrid = xGrid.reshape((len(xGrid), 1))
plt.scatter(X, y, color='red')
plt.plot(xGrid, regressor.predict(xGrid), color='blue')
plt.title('Truth or Bluff Polynomial')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show