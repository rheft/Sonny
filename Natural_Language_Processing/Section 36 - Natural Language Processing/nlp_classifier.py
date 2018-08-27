'''
Create a Bog of Words model.
Run the model through multiple classification algorithms.
Select the best classification algorithm.
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Print breaker
def breaker():
    print("="*50)

# Print results
def print_results(cm, accuracy, error_rate, percision, recall, f1, source):
    print("Results for: {}".format(source))
    print("Confusion matrix: \n{}".format(cm))
    print("Accuracy: {}".format(accuracy))
    print("Error Rate: {}".format(error_rate))
    print("Percision: {}".format(percision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))

# Clean data prior to classifying'
def data_cleaner(data):
    # Clean the text to prepare for models being built
    # We will remove upper case + unneeded puncutation and unneeded words (the, them, an...)
    corpus = []
    ps = PorterStemmer()
    for i in range(0,len(data)):
        review = re.sub('[^a-zA-Z]', ' ', data['Review'][i])                                    # Remove non-alphabetical chars
        review = review.lower()                                                                    # Set all characters to lowercase
        review = review.split()                                                                    # Split the string into character array
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # Remove common words and stem the words into their root ex: loved => love
        review = ' '.join(review)                                                                  # Join the strings back together into one
        corpus.append(review)

    return(corpus)

# Create Bag of Words model
def bag_of_words(corpus):
    # Creating the Bag of Words (BoW) model
    # Creating datatable of words existing in each review (sparse matrix)
    cv = CountVectorizer(max_features = 1500) # Max features leads to less sparcity in the matrix
    X = cv.fit_transform(corpus).toarray()
    return(X)

# Confusion Matrix
def confusion_matrix(y_test, y_pred):
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    tn = cm[0][0]
    tp = cm[1][1]
    fn = cm[0][1]
    fp = cm[1][0]

    accuracy = (tp+tn) / (tn+tp+fp+fn)
    error_rate = 1 - accuracy
    percision = tp / (tp+fp) if tp>0 or fp>0 else 0.0
    recall = tp / (tp+fn) if tp>0 or fn>0 else 0.0
    f1 = 2*(percision*recall)/(percision+recall) if percision>0 or recall>0 else 0.0
    return(cm, accuracy, error_rate, percision, recall, f1)

# Test the classification of each algorithm
def classify(X, y):
    functions = [random_selection, log_regression, naive_bayes, knn, svm, kernel_svm, decision_trees, rand_forest, cart]
    best_f1 = 0
    best_classifier = ""
    # Splitting the Bag of Words into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    breaker() # Initial Break

    for func in functions:
        class_pred, class_source = func(X_train, y_train, X_test, y_test)
        class_cm, class_accuracy, class_error_rate, class_percision, class_recall, class_f1 = confusion_matrix(y_test, class_pred)
        print_results(cm = class_cm, accuracy = class_accuracy, error_rate = class_error_rate, percision = class_percision, recall = class_recall, f1 = class_f1, source = class_source)
        if class_f1 > best_f1:
            best_f1 = class_f1
            best_classifier = class_source
        breaker()

    print("Best classifier: {}\nF1 Score: {}".format(best_classifier, best_f1))

# Define each algorithm
def random_selection(X_train, y_train, X_test, y_test):
    import random
    y_pred = []
    for i in range(0,len(y_test)):
        random_select = 0 if random.random() < 0.5 else 1
        y_pred.append(random_select)

    return(y_pred, "Random")

def naive_bayes(X_train, y_train, X_test, y_test):
    # Fitting Naive Bayes to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return(y_pred, "Naive Bayes")

def log_regression(X_train, y_train, X_test, y_test):
    # Fitting Logistic Regression to the Training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return(y_pred, "Logistic Regression")

def knn(X_train, y_train, X_test, y_test):
    # Fitting K-NN to the Training set
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return(y_pred, "K Nearest Neighbor")

def svm(X_train, y_train, X_test, y_test):
    # Fitting SVM to the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return(y_pred, "Linear SVM")

def kernel_svm(X_train, y_train, X_test, y_test):
    # Fitting Kernel SVM to the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return(y_pred, "Kernel SVM")

def decision_trees(X_train, y_train, X_test, y_test):
    # Fitting Decision Tree Classification to the Training set
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion = 'entropy')
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return(y_pred, "Decision Trees")

def rand_forest(X_train, y_train, X_test, y_test):
    # Fitting Random Forest Classification to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return(y_pred, "Random Forest")

def cart(X_train, y_train, X_test, y_test):
    # Fitting Decision Tree Classification to the Training set
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return(y_pred, "CART")

def maxent(X_train, y_train, X_test, y_test):
    algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
    classifier = nltk.MaxentClassifier.train(X_train, algorithm)
    classifier.show_most_informative_features(10)

    y_pred = []
    for row in X_test:
        determined_label = classifier.classify(row)
        y_pred.append(determined_label)

    return(y_pred, "Max Entropy")

if __name__ == "__main__":
    # Import the data set
    dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

    # Clean data
    corpus = data_cleaner(dataset)

    # Create bag of words model
    features = bag_of_words(corpus)
    target = dataset.iloc[:,1]

    # Create and print results of classification models
    classify(X = features, y = target)
