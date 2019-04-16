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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

# Import data
dataset = LoadData("Wine.csv").data

# Split the dataset
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

# Lets do some preprocessing...
processor = PreProcessing()
# Split the data
X_train, X_test, y_train, y_test = processor.split(X, y, test_size=0.2)
# scale the data
X_train = processor.fit_scaler(X_train)
X_test = processor.scale(X_test)

# Apply PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Lets fit the model now
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predict!
y_pred = classifier.predict(X_test)

# Creating the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Fine, lets visualize it.. I geuss its more fun 🤷‍
visual = ClassifierVisual(X_train, y_train, classifier)
visual.visualize(title='Logistic Regression', xlab='Age', ylab='Salary', colors=('red', 'green', 'blue'))
