# Imports
import sys
import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Import Deep Learning libraries (keras)
import keras
from keras.models import Sequential
from keras.layers import Dense

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

# Apply feature scaling
X_train = processor.fit_scaler(X_train)
X_test = processor.fit_scaler(X_test)

# Initialize the Artificial Neural Network (ANN)
classifier = Sequential()
# Create the input and first hidden layers
classifier.add(Dense(input_dim=11, activation='relu', units=8, kernel_initializer='uniform'))
# Create the second hidden layer
classifier.add(Dense(activation='relu', units=8, kernel_initializer='uniform'))
# Create the output layer
classifier.add(Dense(activation='sigmoid', units=1, kernel_initializer='uniform'))

# Compile ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Lets Fit the ANN now...
classifier.fit(X_train, y_train, batch_size=5, nb_epoch=250)

# Predicting the test results
y_pred = classifier.predict(X_test)
y_pred_values = [1 if y > 0.5 else 0 for y in y_pred]

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_values)
cm
