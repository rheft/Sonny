# import dataset
source('config.R')
dataset = read.csv(paste0(data_locale,'Ads_CTR_Optimisation.csv'))

# UCB Selection
N = 10000
d = 10
ads_selected = integer()
Ni_1 <- integer(d)
Ni_0 <- integer(d)
total_reward = 0
for(n in 1:N){
  rand_max = 0
  ad = 0
  for (i in 1:d){
    rand_beta = rbeta(n = 1, shape1 = Ni_1[i] + 1, shape2 = Ni_0[i] + 1)
    if(rand_beta > rand_max){
      rand_max = rand_beta
      ad = i
    }
  }
  ads_selected = append(ads_selected, ad)
  reward = dataset[n, ad]
  if(reward == 1) {
    Ni_1[ad] = Ni_1[ad] + 1
  }
  else {
    Ni_0[ad] = Ni_0[ad] + 1
  }
  total_reward = total_reward + reward
}

total_reward

##### Random Selection
N = 10000
d = 10
ads_selected = integer(0)
total_reward = 0
for (n in 1:N) {
  ad = sample(1:10, 1)
  ads_selected = append(ads_selected, ad)
  reward = dataset[n, ad]
  total_reward = total_reward + reward
}

ads_selected
total_reward

# Visualising the results
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')
