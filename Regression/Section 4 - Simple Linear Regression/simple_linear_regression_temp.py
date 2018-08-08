#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 06:16:44 2018

@author: robheft
"""

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the data set
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values # Features
y = dataset.iloc[:, 1].values # Target

# Split the dataset into training/testing sets
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

# Fitting the linear regression model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict values from the test set
y_pred = regressor.predict(x_test)

# Visualize the Training results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary and Experiance (Training Set)')
plt.xlabel('Years of experiance')
plt.ylabel('Salary')
plt.show

# Visualize the test results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary and Experiance (Test Set)')
plt.xlabel('Years of experiance')
plt.ylabel('Salary')
plt.show
