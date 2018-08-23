# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Import dataset
data_locale = '/Users/rheft/Documents/Sonny/Data_Sets/'
dataset = pd.read_csv(data_locale+'Ads_CTR_Optimisation.csv')
dataset

# UCB selection
N = 10000 # number of data points
d = 10 # number of ads
ads_selected = []
Ni = [0] * d
Ri = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if(Ni[i] > 0):
            ri = Ri[i] / Ni[i]
            delta_i = math.sqrt(1.5 * math.log(n + 1) / Ni[i])
            upper_bound = ri + delta_i
        else:
            upper_bound = 1e400
        if(upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ad = i

    ads_selected.append(ad)
    Ni[ad] = Ni[ad] + 1
    reward = dataset.values[n, ad]
    Ri[ad] = Ri[ad] + reward
    total_reward = total_reward + reward

ads_selected

plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()


# What random selection would look like
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward

total_reward

plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
