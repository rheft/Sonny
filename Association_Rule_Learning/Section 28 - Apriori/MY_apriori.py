# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from apyori import apriori

# Import data set
data_locale = '/Users/rheft/Documents/Sonny/Data_Sets/'
dataset = pd.read_csv(data_locale+'Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
transactions

# Train the Apriori model
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
rules

# Visualize the results
results = list(rules)
results[1:5]
