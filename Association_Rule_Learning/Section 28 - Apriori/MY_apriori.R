# Apriori Association Rule Learning
library(arules)
source('config.R', local = TRUE)

# Data preprocessing
dataset = read.transactions(paste0(data_locale, 'Market_Basket_Optimisation.csv'), sep = ",", rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training apriori
rules <- apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Visualizing the Apriori
inspect(sort(rules, by = 'lift')[1:10])
