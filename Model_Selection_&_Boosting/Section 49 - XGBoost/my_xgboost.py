### Deep Learning ###

# Importing the libraries
import sys
sys.path.insert(0, '/Users/rheft/dev/Sonny/')
from config import data_locale
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifierhyd

# Importing the dataset
dataset = pd.read_csv(data_locale+'Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values # Features/independant vars
y = dataset.iloc[:, 13].values # Target/dependant vars

# Encode categorical variables
labelencoder_X_country = LabelEncoder()
X[:, 1] = labelencoder_X_country.fit_transform(X[:, 1])
labelencoder_X_gender = LabelEncoder()
X[:, 2] = labelencoder_X_gender.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Applying XGBoost
classifier = XGBClassifier()
