#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 06:14:15 2018

@author: robheft
"""

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import data
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values

# Dendrogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

# fitting hc to dataset
from sklearn.cluster import AgglomerativeClustering as ac
hc = ac(n_clusters = 5,
        affinity = 'euclidean',
        linkage = 'ward')
y_hc = hc.fit_predict(x)

# Visualizing clusters
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, color = 'red', label = 'Careful')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, color = 'blue', label = 'Standard')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, color = 'green', label = 'Target')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, color = 'cyan', label = 'Careless')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, color = 'magenta', label = 'Sensible')
plt.title('Clusters of clients')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()





