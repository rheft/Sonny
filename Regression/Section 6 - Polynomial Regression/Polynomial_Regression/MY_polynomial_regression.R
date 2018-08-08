# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset <- dataset[, 2:3]

# # Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)
# 
# # Feature Scaling
# # training_set = scale(training_set)
# # test_set = scale(test_set)

# Fitting Linear model
linReg <- lm(formula = Salary~., 
             data = dataset)

# Fitting Polynomial model
dataset$Level2 <- dataset$Level^2
dataset$Level3 <- dataset$Level^3
dataset$Level4 <- dataset$Level^4
dataset$Level5 <- dataset$Level^5
dataset$Level6 <- dataset$Level^6

polyReg <- lm(formula = Salary~., 
              data = dataset)

# Visualize the regression models
library(ggplot2)

# Comparison
ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary),
             color='red') +
  geom_line(aes(x=dataset$Level, y=predict(linReg, newdata = dataset)),
            color='blue') +
  geom_line(aes(x=dataset$Level, y=predict(polyReg, newdata = dataset)),
            color='green') +
  ggtitle('Truth or Bluff? Poly.') +
  xlab('Level') +
  ylab('Salary')

# Predict
# Linear
y_pred = predict(linReg, newdata = data.frame(Level = 6.5)) #= $330k
y_poly_pred = predict(polyReg, newdata = data.frame(Level = 6.5, 
                                                   Level2 = 6.5^2,
                                                   Level3 = 6.5^3,
                                                   Level4 = 6.5^4,
                                                   Level5 = 6.5^5,
                                                   Level6 = 6.5^6))
