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
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

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
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

# Predict!
y_pred = classifier.predict(X_test)

# Creating the confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm

# Implementing k-fold cross validaiton
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
avg = accuracies.mean()
std_dev_acc = accuracies.std()
avg
std_dev_acc

# Applying grid search
parameters = [
    {"C": [1.0, 10, 100, 1000], "kernel": ["linear"]},
    {"C": [1.0, 10, 100, 1000], "kernel": ["rbf"], "gamma":[0.3, 0.4, 0.5, 0.6, 0.7]}
]
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)

grid_search.best_score_
grid_search.best_params_
# Fine, lets visualize it.. I geuss its more fun 🤷‍
visual = ClassifierVisual(X_train, y_train, classifier)
visual.visualize(title='Linear SVM', xlab='Age', ylab='Salary')
