import sys
import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Import lib files
envs = load_dotenv(find_dotenv())
file = os.getenv("lib")
sys.path.insert(0, file)
from utils import LoadData
from preprocessing import PreProcessing
from visuals import ClassifierVisual

# Import model library
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Import data
dataset = LoadData("Social_Network_Ads.csv").data

# Split the dataset
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Lets do some preprocessing...
processor = PreProcessing()
# Split the data
X_train, X_test, y_train, y_test = processor.split(X, y, test_size=0.25)
# scale the data
X_train = processor.fit_scaler(X_train)
X_test = processor.scale(X_test)

# Lets fit the model now
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

# Predict!
y_pred = classifier.predict(X_test)

# Creating the confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm
# Fine, lets visualize it.. I geuss its more fun 🤷‍
visual = ClassifierVisual(X_test, y_test, classifier)
visual.visualize(title='K Nearest Neighbor', xlab='Age', ylab='Salary')
