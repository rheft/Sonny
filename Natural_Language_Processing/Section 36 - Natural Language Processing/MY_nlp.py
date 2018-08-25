# Natural Language Processing

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
# nltk.download('stopwords') # Download stopwords (common words)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from config import data_locale
data_locale

# Import the data set
dataset = pd.read_csv(data_locale+'Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Clean the text to prepare for models being built
# We will remove upper case + unneeded puncutation and unneeded words (the, them, an...)
corpus = []
ps = PorterStemmer()
for i in range(0,len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])                                    # Remove non-alphabetical chars
    review = review.lower()                                                                    # Set all characters to lowercase
    review = review.split()                                                                    # Split the string into character array
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # Remove common words and stem the words into their root ex: loved => love
    review = ' '.join(review)                                                                  # Join the strings back together into one
    corpus.append(review)

# Creating the Bag of Words (BoW) model
# Creating datatable of words existing in each review (sparse matrix)
cv = CountVectorizer(max_features = 1500) # Max features leads to less sparcity in the matrix
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1]

# Classifying the NLP w/Naive Bayes
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
error_rate = 1-accuracy
accuracy
error_rate
