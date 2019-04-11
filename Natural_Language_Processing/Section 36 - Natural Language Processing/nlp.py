#NLP
import sys
import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import math
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
# Import lib files
envs = load_dotenv(find_dotenv())
file = os.getenv("lib")
sys.path.insert(0, file)
from utils import LoadData
from preprocessing import PreProcessing
from visuals import ClassifierVisual
from sklearn.metrics import confusion_matrix
# Import Data
dataset = LoadData("Restaurant_Reviews.tsv", seperator='\t').data

### Cleaning the texts ###
ps = PorterStemmer()
corpus = []
for i in range(0, dataset.shape[0]):
    review = dataset['Review'][i]
    review = re.sub(pattern='[^a-zA-Z\s]', repl='', string=review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = " ".join(review)
    corpus.append(review)

corpus
dataset['cleaned'] = corpus

# Create the Bag of Words model BoW
cv = CountVectorizer(max_features = 500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1]

### Build a model using the BoW model
# Lets do some preprocessing...
processor = PreProcessing()
# Split the data
X_train, X_test, y_train, y_test = processor.split(X, y, test_size=0.20)

# Lets fit the model now
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predict!
y_pred = classifier.predict(X_test)

# Creating the confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm
