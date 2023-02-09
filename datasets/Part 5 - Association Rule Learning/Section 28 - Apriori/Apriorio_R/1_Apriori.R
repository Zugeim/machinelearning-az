# A PRIORI

# Importar el data set
install.packages("arules")
install.packages("arulesViz")
library(arules)
dataset = read.csv("Market_Basket_Optimisation.csv", header = FALSE)
dataset = read.transactions("Market_Basket_Optimisation.csv",
                            sep = ",",
                            rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, 
                  topN = 100)


# Entrenar algoritmo Apriori con el dataset
rules = apriori(data = dataset,
                parameter = list(support = 0.003, # 3*7/7500 redondeado
                                 confidence = 0.2)) # Empezar con una no demasiado alta
                                                    # ni demasiado baja e ir jugando con
                                                    # el parametro.

# Visualización de los resultados
inspect(sort(rules, by = 'lift')[1:10])
#install.packages("arulesViz")
library(arulesViz)
plot(rules, method = "graph", engine = "htmlwidget")
