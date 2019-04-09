import sys
import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import math

# Import lib files
envs = load_dotenv(find_dotenv())
file = os.getenv("lib")
sys.path.insert(0, file)
from utils import LoadData
from preprocessing import PreProcessing
from visuals import ClassifierVisual

# Import Data
dataset = LoadData("Ads_CTR_Optimisation.csv").data

# Implementing Random Selection
N = 10000
d = 10
ads_selected = []
random_total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    random_total_reward = random_total_reward + reward
random_total_reward

#Visualize the random
plt.hist(ads_selected)
plt.title('Histogram of RANDOM ds selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

# Implementing Thompson Sampling algorithm
ads_selected = []
number_of_rewards_1 = [0]*d
number_of_rewards_0 = [0]*d
ts_total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        number_of_rewards_1[ad] += 1
    else:
        number_of_rewards_0[ad] += 1
    ts_total_reward = ts_total_reward + reward
ts_total_reward
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of Thompson Sampling ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

# Implementing UCB algorithm
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
ucb_total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    ucb_total_reward = ucb_total_reward + reward
ucb_total_reward
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of UCB ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
