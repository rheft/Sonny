# Apriori

# Importing the libraries
import sys
import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Import lib files
envs = load_dotenv(find_dotenv(usecwd=True))
file = os.getenv("lib")
sys.path.insert(0, file)
print(file)
from utils import LoadData
from preprocessing import PreProcessing
from visuals import ClassifierVisual

# Import the apriori classifier
from apyori import apriori

# Data Preprocessing
dataset = LoadData("Market_Basket_Optimisation.csv", header=None).data

# Get transactions
transactions = []
for i in range(0,dataset.shape[0]):
    transactions.append([str(dataset.values[i,j]) for j in range(0,dataset.shape[1])])

# create the model
rules = apriori(
    transactions,
    min_support=0.003,
    min_confidence=0.2,
    min_lift=3,
    min_length=2
)

# Visualize!
results = list(rules)
results
