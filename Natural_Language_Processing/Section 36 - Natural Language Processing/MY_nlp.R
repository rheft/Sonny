# Natural Language Processing

# Import the dataset
source("/Users/rheft/dev/Sonny/config.R")
dataset_og = read.delim(paste0(data_locale,'Restaurant_Reviews.tsv'), quote='')

# Clean the text
library('tm')
library('SnowballC') # Used for stopwords
corpus = VCorpus(VectorSource(dataset_og$Review)) # Create corpus
corpus = tm_map(corpus, content_transformer(tolower)) # Convert all letters to lowercase
corpus = tm_map(corpus, removeNumbers) # Remove any numbers
corpus = tm_map(corpus, removePunctuation) # Remove any punctuations
corpus = tm_map(corpus, removeWords, stopwords()) # Remove unneeded words (and, this, the etc.)
corpus = tm_map(corpus, stemDocument) # take just root of each word
corpus = tm_map(corpus, stripWhitespace) # Remove whitespace

# Create the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_og$Liked

# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
set.seed(123)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
cm
