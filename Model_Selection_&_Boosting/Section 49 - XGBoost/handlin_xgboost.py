# Imports
import sys
import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Import Requierd libraries
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

# Import lib files
envs = load_dotenv(find_dotenv())
file = os.getenv("lib")
sys.path.insert(0, file)
from utils import LoadData
from preprocessing import PreProcessing
from visuals import ClassifierVisual

# Import data
dataset = LoadData("Churn_Modelling.csv").data
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Lets do some preprocessing...
processor = PreProcessing()
# Encode the data (Country/Gender)
X[:, 1] = processor.encode(X[:, 1])
X[:, 2] = processor.encode(X[:, 2])
X = processor.hot_encoding(data = X, features=[1])
X = X[:, 1:]

# Split the data into training+test
X_train, X_test, y_train, y_test = processor.split(X, y, test_size=0.2)

# Fitting XGboost
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the test results
y_pred = classifier.predict(X_test)

# Making the confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm

# apply k-fold cross validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()
