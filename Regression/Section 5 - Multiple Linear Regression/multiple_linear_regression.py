import sys
import os
from dotenv import load_dotenv, find_dotenv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

# Import lib files
envs = load_dotenv(find_dotenv())
file = os.getenv("lib")
sys.path.insert(0, file)
from utils import LoadData
from preprocessing import PreProcessing

# Load data
dataset = LoadData("50_Startups.csv").data

# Split the dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Using the PreProcessing class from preprocessing
processor = PreProcessing()
# Encoding dummy variables
X = processor.dummy_encoding(data=X, feature_position=3)

# Avoiding the dummy variable trap
X = X[:, 1:]

# Building the optimal model using Backward Elimination
X = np.append(arr=np.ones((X.shape[0], 1)).astype(int), values=X, axis=1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X = X[:, 3]

# Split the data
X_train, X_test, y_train, y_test = processor.split(X, y, test_size=0.2, random_state=0)

# Fit multiple linear regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Visualizing the data
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs. Exp.')
plt.xlabel('Years of Experiance')
plt.ylabel('Salary')
plt.show()
