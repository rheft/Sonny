# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Import dataset
data_locale = '/Users/rheft/Documents/Sonny/Data_Sets/'
dataset = pd.read_csv(data_locale+'Ads_CTR_Optimisation.csv')
dataset

# Thompson Sampling selection
N = 10000 # number of data points
d = 10 # number of ads
ads_selected = []
Ni_1 = [0] * d
Ni_0 = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        rand_beta = random.betavariate(Ni_1[i] + 1, Ni_0[i] + 1)
        if(rand_beta > max_random):
            max_random = rand_beta
            ad = i

    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        Ni_1[ad] = Ni_1[ad] + 1
    else:
        Ni_0[ad] = Ni_0[ad] + 1
    total_reward = total_reward + reward

ads_selected
total_reward

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
