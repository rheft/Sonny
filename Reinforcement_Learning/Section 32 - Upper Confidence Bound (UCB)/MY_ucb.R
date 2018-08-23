# import dataset
source('config.R')
dataset = read.csv(paste0(data_locale,'Ads_CTR_Optimisation.csv'))

# UCB Selection
N = 10000
d = 10
ads_selected = integer()
Ni <- integer(d)
Ri <- integer(d)
total_reward = 0
for(n in 1:N){
  max_ucb = 0
  ad = 0
  for (i in 1:d){
    if(Ni[i] > 0){
      ri = Ri[i]/Ni[i]
      delta_i = sqrt(1.5*log(n)/Ni[i])
      ucb = ri + delta_i
    }
    else{
      ucb = 1e400
    }
    if(ucb > max_ucb){
      max_ucb = ucb
      ad = i
    }
  }
  ads_selected = append(ads_selected, ad)
  Ni[ad] = Ni[ad] + 1
  reward = dataset[n, ad]
  Ri[ad] = Ri[ad] + reward
  total_reward = total_reward + reward
}


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
