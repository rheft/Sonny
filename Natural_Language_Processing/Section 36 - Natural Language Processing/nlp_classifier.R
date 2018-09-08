# NLP Classifiers

# Import Libraries
library(tm) # Used for sparce matrix manipulation
library(SnowballC) # Used for stopwords
library(caTools)
library(randomForest)
library(rpart)
library(e1071)
library(class)
library(C50)
library(maxent)

getCorpus <- function(data){
  corpus = VCorpus(VectorSource(data$Review)) # Create corpus
  corpus = tm_map(corpus, content_transformer(tolower)) # Convert all letters to lowercase
  corpus = tm_map(corpus, removeNumbers) # Remove any numbers
  corpus = tm_map(corpus, removePunctuation) # Remove any punctuations
  corpus = tm_map(corpus, removeWords, stopwords()) # Remove unneeded words (and, this, the etc.)
  corpus = tm_map(corpus, stemDocument) # take just root of each word
  corpus = tm_map(corpus, stripWhitespace) # Remove whitespace

  return(corpus)
}

bagOfWords <- function(corpus){
  # Create the Bag of Words model
  dtm = DocumentTermMatrix(corpus)
  dtm = removeSparseTerms(dtm, 0.999)
  dataset = as.data.frame(as.matrix(dtm))

  return(dataset)
}

breaker <- function() {
  cat(paste0(rep("=", 50), sep = ""))
  cat(paste0("\n"))
}

confusionMatrix <- function(y_pred, y_actual){
  # Making the Confusion Matrix
  cm = table(y_actual, y_pred)
  tn = cm[1]
  fp = cm[2]
  fn = cm[3]
  tp = cm[4]

  # Calculate other metrics
  accuracy = (tp+tn) / (tn+tp+fp+fn)
  error_rate = 1 - accuracy
  if(tp>0 || fp>0) percision = tp / (tp+fp)  else 0.0
  if(tp>0 || fn>0) recall = tp / (tp+fn) else 0.0
  if(percision>0 || recall>0) f1 = 2*(percision*recall)/(percision+recall) else 0.0

  return(list(cm, accuracy, error_rate, percision, recall, f1))
}

printResults <- function(results, source){

  # Print results
  cat(paste("Results for", source, "\n"))
  cat(paste("Confusion Matrix:", results[1], "\n"))
  cat(paste("Accuracy:", results[2], "\n"))
  cat(paste("Error Rate:", results[3], "\n"))
  cat(paste("Percision:", results[4], "\n"))
  cat(paste("Recall:", results[5], "\n"))
  cat(paste("F1 Score:", results[6], "\n"))
  breaker()
}

classifyDataset <- function(dataset){
  # Splitting the dataset into the Training set and Test set
  # install.packages('caTools')
  library(caTools)
  set.seed(123)
  split = sample.split(dataset$Liked, SplitRatio = 0.8)
  training_set = subset(dataset, split == TRUE)
  test_set = subset(dataset, split == FALSE)
  y_actual = test_set[, 692]

  # Initial line break
  breaker()

  # Define funcion list
  func_list <- list(randForest,
                    decsionTrees,
                    naiveBayesClassifier,
                    kernelSVM,
                    linearSVM,
                    knnClassifier,
                    logRegression,
                    cFiveZero,
                    maxEntropy)

  # Execute each function
  best_class = ""
  best_score = 0
  for(func in func_list){
    func_results = func(training_set, test_set)
    cm_results = confusionMatrix(func_results$predictions, y_actual)
    # cat(as.character(cm_results))
    func_f1 = as.numeric(cm_results[6])
    if(func_f1 > best_score){
      best_score = func_f1
      best_class = func_results$source
    }
    printResults(cm_results, func_results$source)
  }

  breaker()
  cat(paste("\nBest Results:\n", "Class:", best_class, "\nScore:", best_score, "\n"))
}

randForest <- function(training_set, test_set){
  # Fitting Random Forest Classification to the Training set
  classifier = randomForest(x = training_set[-692], y = training_set$Liked, ntree = 10)

  # Predicting the Test set results
  y_pred = predict(classifier, newdata = test_set[-692])
  return(list("predictions" = y_pred, "source" = "Random Forest"))
}

decsionTrees <- function(training_set, test_set){
  # Fitting Decision Tree Classification to the Training set
  classifier = rpart(formula = Liked ~ .,
                     data = training_set)

  # Predicting the Test set results
  y_pred = predict(classifier, newdata = test_set[-692], type = 'class')
  return(list("predictions" = y_pred, "source" = "Decision Trees"))
}

naiveBayesClassifier <- function(training_set, test_set){
  classifier = naiveBayes(x = training_set[-692], y = training_set$Liked)

  # Predicting the Test set results
  y_pred = predict(classifier, newdata = test_set[-692])
  return(list("predictions" = y_pred, "source" = "Naive Bayes"))
}

kernelSVM <- function(training_set, test_set){
  classifier = svm(formula = Liked ~ .,
                   data = training_set,
                   type = 'C-classification',
                   kernel = 'radial')

  # Predicting the Test set results
  y_pred = predict(classifier, newdata = test_set[-692])
  return(list("predictions" = y_pred, "source" = "Kernel SVM"))
}

linearSVM <- function(training_set, test_set){
  classifier = svm(formula = Liked ~ .,
                   data = training_set,
                   type = 'C-classification',
                   kernel = 'linear')

  # Predicting the Test set results
  y_pred = predict(classifier, newdata = test_set[-692])
  return(list("predictions" = y_pred, "source" = "Linear SVM"))
}

knnClassifier <- function(training_set, test_set){
  y_pred = knn(train = training_set[, -692],
               test = test_set[, -692],
               cl = training_set[, 692],
               k = 5,
               prob = TRUE)
  return(list("predictions" = y_pred, "source" = "k Nearest Neighbor"))
}

logRegression <- function(training_set, test_set){
  classifier = glm(formula = Liked ~ .,
                   family = binomial,
                   data = training_set)

  # Predicting the Test set results
  prob_pred = predict(classifier, type = 'response', newdata = test_set[-692])
  y_pred = ifelse(prob_pred > 0.5, 1, 0)
  return(list("predictions" = y_pred, "source" = "Logistic regression"))
}

cFiveZero <- function(training_set, test_set){
  classifier = C5.0(formula = Liked ~ .,
                    data = training_set,
                    rules = TRUE)
  y_pred = predict(classifier, newdata = test_set[-692])
  return(list("predictions" = y_pred, "source" = "C5.0"))
}

maxEntropy <- function(training_set, test_set){
  classifier = maxent(training_set[,-692], training_set[,692])

  y_pred = predict(classifier, test_set[,-692])
  y_pred = y_pred[, "labels"]
  return(list("predictions" = y_pred, "source" = "Maximum Entropy"))
}

main <- function() {
  # Import the dataset
  source("/Users/rheft/dev/Sonny/config.R")
  dataset_og = read.delim(paste0(data_locale,'Restaurant_Reviews.tsv'), quote='')

  corpus = getCorpus(dataset_og)
  bow_data = bagOfWords(corpus)
  bow_data$Liked = dataset_og$Liked

  # Encoding the target feature as factor
  bow_data$Liked = factor(bow_data$Liked, levels = c(0, 1))

  # Classify the dataset using each classifier
  classifyDataset(bow_data)
}

main()
