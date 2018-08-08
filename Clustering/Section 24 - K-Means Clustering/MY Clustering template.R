# Import dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[,4:5]

# Use elbow method to find optimal # of cluster
set.seed(6)
wcss <- vector()
for (i in 1:10) wcss[i] = sum(kmeans(X, i)$withinss)
plot(1:10,
     wcss,
     type = 'b',
     main = paste('clusters of clients'),
     xlab = 'Number of clusters',
     ylab = 'WCSS')

# Cluster based on elbow = 5
set.seed(29)
kmeans <- kmeans(X, 5, iter.max = 300, nstart = 10)

# Visualize the clusters
library(cluster)
clusplot(X,
         kmeans$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste("Clusters of clients"),
         xlab = "Annual income",
         ylab = 'Spending score')


