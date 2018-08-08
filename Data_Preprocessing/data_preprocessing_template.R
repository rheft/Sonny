# Data Preprocessing

# Import Data
dataset <- read.csv('Data.csv')
#dataset <- dataset[, 2:3]

# Split dataset into training/testing sets
library(caTools)
set.seed(123)
split <- sample.split(dataset$Purchased, SplitRatio = 0.8)
trainingset <- subset(dataset, split == TRUE)
testingset <- subset(dataset, split == FALSE)

# Feature Scaling
# trainingset[, 2:3] <- scale(trainingset[, 2:3])
# testingset[, 2:3] <- scale(testingset[, 2:3])
