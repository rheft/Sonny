data_local = "/Users/rheft/Documents/Sonny/Data_Sets/"
# Import Data sets
dataset = read.csv(paste0(data_local,"Mall_Customers.csv"))
x = dataset[,c(4,5)]
