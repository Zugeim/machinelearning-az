# ECLAT


# Importar el data set
#install.packages("arules")
library(arules)
dataset = read.csv("Market_Basket_Optimisation.csv", header = FALSE)
dataset = read.transactions("Market_Basket_Optimisation.csv",
                            sep = ",",
                            rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, 
                  topN = 100)


# Entrenar algoritmo Apriori con el dataset
rules = eclat(data = dataset,
                parameter = list(support = 0.003,
                                 minlen = 2))

# Visualización de los resultados
inspect(sort(rules, by = 'support')[1:10])
#install.packages("arulesViz")
library(arulesViz)
plot(rules, method = "graph", engine = "htmlwidget")
