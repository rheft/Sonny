# Imports
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Preprocessing class
class PreProcessing():
    def __init__(self):
        print("Created preprocessor")

    def _replace(self, data, strategy):
        imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
        imputer = imputer.fit(data)
        return imputer.transform(data)

    def replace_with_mean(self, data):
        return self._replace(data, "mean")

    def replace_with_median(self, data):
        return self._replace(data, "median")

    def replace_with_frequent(self, data):
        return self._replace(data, "most_frequent")

    def replace_with_predicted(self, data):
        print("Need to create this yet, sorrrrryyyyy")
        return data

    def encode(self, data):
        encoder = LabelEncoder()
        return encoder.fit_transform(data)

    def hot_encoding(self, data, features):
        hot_encoder = OneHotEncoder(categorical_features=features)
        return hot_encoder.fit_transform(data).toarray()

    def dummy_encoding(self, data, feature_position):
        data[:, feature_position] = self.encode(data[:, feature_position])
        data = self.hot_encoding(data, [feature_position])
        return data

    def scale(self, data):
        return self._scaler.transform(data)

    def fit_scaler(self, data):
        self._scaler = StandardScaler()
        self._scaler.fit(data)
        return self.scale(data)

    def split(self, features, target, test_size=0.2, random_state=0):
        return train_test_split(features, target, test_size=test_size, random_state=random_state)
