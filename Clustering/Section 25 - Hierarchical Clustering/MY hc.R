data_local = "/Users/rheft/Documents/Sonny/Data_Sets/"
# Import Data sets
dataset = read.csv(paste0(data_local,"Mall_Customers.csv"))
x = dataset[4:5]

# Creating the dendrogram
dendo = hclust(dist(x, method = 'euclidean'),method = 'ward.D')
plot(dendo, main = paste('Dendrogram'), xlab = 'Customers', ylab = 'Euclidean Distance')

# Fit the hierarchical clustering
hc = hclust(dist(x, method = 'euclidean'),method = 'ward.D')
y_hc = cutree(hc, k = 5)

# Visualize the clusters
library(cluster)
clusplot(,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste("Clusters of clients"),
         xlab = "Annual income",
         ylab = 'Spending score')
