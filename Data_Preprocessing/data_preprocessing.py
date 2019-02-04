# Normal Imports
import sys
import os
from dotenv import load_dotenv, find_dotenv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
from sklearn.model_selection import train_test_split

# Import lib files
envs = load_dotenv(find_dotenv())
file = os.getenv("lib")
sys.path.insert(0, file)
from utils import LoadData
from preprocessing import PreProcessing

# Load the data
dataset = LoadData("Data.csv").data

# Split the dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Handling missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
X[:, 1:3]

# Handling Categorical Variables
encoder_X = LabelEncoder()
one_hot_encoder = OneHotEncoder(categorical_features = [0])
X[:, 0] = encoder_X.fit_transform(X[:, 0])
X = one_hot_encoder.fit_transform(X).toarray()

# We need to encode the dependant variable as well
encoder_y = LabelEncoder()
y = encoder_X.fit_transform(y)
np.around(X, decimals=0).astype(int)

# Splitting the data into train adn test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#-----------Using the PreProcessing class from preprocessing.py:---------------
processor = PreProcessing()
# Fill Missing data
X[:, 1:3] = processor.replace_with_mean(X[:, 1:3])
# Handle categorical data
X[:, 0] = processor.encode(X[:, 0])
X = processor.hot_encoding(X, features=[0])
# Encode target
y = processor.encode(y)
# Split the data
X_train, X_test, y_train, y_test = processor.split(X, y, test_size=0.2, random_state=0)
# Feature Scaling
X_train = processor.fit_scaler(X_train)
X_test = processor.scale(X_test)
