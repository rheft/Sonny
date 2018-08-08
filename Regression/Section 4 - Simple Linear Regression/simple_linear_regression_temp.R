# Import Data
dataset <- read.csv('Salary_Data.csv')

# Split dataset into training/testing sets
library(caTools)
library(ggplot2)
set.seed(123)
split <- sample.split(dataset$Salary, SplitRatio = 2/3)
trainingset <- subset(dataset, split == TRUE)
testingset <- subset(dataset, split == FALSE)

# Simple Linear Regression
# Fitting simple linear regression
regressor = lm(formula = Salary ~ YearsExperience, trainingset)

# Predict
y_pred = predict(regressor, newdata = testingset)

# Visualize the test set results
ggplot() + 
  geom_point(aes(x=trainingset$YearsExperience, y=trainingset$Salary),color='red') +
  geom_line(aes(x=trainingset$YearsExperience, y=predict(regressor, newdata = trainingset)), color='blue')+
  ggtitle('Salary vs Experiance (Training set)')+
  xlab('Years of experiance')+
  ylab('Salary')

# Visualize the test set results
ggplot() + 
  geom_point(aes(x=testingset$YearsExperience, y=testingset$Salary),color='red') +
  geom_line(aes(x=trainingset$YearsExperience, y=predict(regressor, newdata = trainingset)), color='blue')+
  ggtitle('Salary vs Experiance (Test set)')+
  xlab('Years of experiance')+
  ylab('Salary')