# Normal Imports
import sys
import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

# Import lib files
envs = load_dotenv(find_dotenv())
file = os.getenv("lib")
sys.path.insert(0, file)
from utils import LoadData
from preprocessing import PreProcessing

# Load the data
dataset = LoadData("Salary_Data.csv").data

# Split the dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Using the PreProcessing class from preprocessing
processor = PreProcessing()

# Split the data
X_train, X_test, y_train, y_test = processor.split(X, y, test_size=0.2, random_state=0)

# Fit Simple Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the test set
y_pred = regressor.predict(X_test)

# Visualizing the data
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs. Exp.')
plt.xlabel('Years of Experiance')
plt.ylabel('Salary')
plt.show()
