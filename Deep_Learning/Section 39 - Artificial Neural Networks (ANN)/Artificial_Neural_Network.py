### Deep Learning ###

# Importing the libraries
import sys
sys.path.insert(0, '/Users/rheft/dev/Sonny/')
from config import data_locale
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def main():
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
    X

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    ### Making the Artificial Neural Network

    # Initialize the keras ann
    classifier = Sequential()
    # Add input layer and first hidden layer
    classifier.add(layer=Dense(activation='relu', input_dim=11, units=6, kernel_initializer='uniform', ))
    # Add another hidden layer
    classifier.add(layer=Dense(activation='relu', units=6, kernel_initializer='uniform'))
    # Add the output layer
    classifier.add(layer=Dense(activation='sigmoid', units=1, kernel_initializer='uniform', ))
    # Compile the ANN!!
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Lets play.
    classifier.fit(X_train, y_train, batch_size=10, epochs=100)

    ### Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred>0.5)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

if __name__ == "__main__":
    main()
