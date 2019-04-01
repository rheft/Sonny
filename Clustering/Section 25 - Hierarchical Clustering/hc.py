# K-Means Clustering

# Importing the libraries
import sys
import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Import machine learning package
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Import lib files
envs = load_dotenv(find_dotenv())
file = os.getenv("lib")
sys.path.insert(0, file)
from utils import LoadData
from preprocessing import PreProcessing

# Importing the dataset
dataset = LoadData('Mall_Customers.csv').data
X = dataset.iloc[:, [3, 4]].values

# use dendogram to clusters
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.xlabel('Euclidean Distance')
plt.ylabel('Customers')
plt.title('Dendrogram')
plt.show()

# Fitting kmeans
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Visualizations
plt.scatter(X[y_hc==0, 0], X[y_hc == 0, 1], s=100, c='red', label='Careful')
plt.scatter(X[y_hc==1, 0], X[y_hc == 1, 1], s=100, c='blue', label='Standard')
plt.scatter(X[y_hc==2, 0], X[y_hc == 2, 1], s=100, c='green', label='Targets')
plt.scatter(X[y_hc==3, 0], X[y_hc == 3, 1], s=100, c='cyan', label='Careless')
plt.scatter(X[y_hc==4, 0], X[y_hc == 4, 1], s=100, c='magenta', label='Sensible')
plt.xlabel('Annual  income')
plt.ylabel('spending score')
plt.title('clusters of clients')
plt.legend()
plt.show()
