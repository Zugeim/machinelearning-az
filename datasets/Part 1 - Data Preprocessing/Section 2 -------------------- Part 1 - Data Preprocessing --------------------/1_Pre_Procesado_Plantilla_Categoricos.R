# Plantilla para el Pre Procesado de Datos Categóricos

# Importar el dataset
dataset <- read.csv('Data.csv')

# Codificar las variables categóricas
dataset$Country <- factor(dataset$Country,
                          levels = c("France", "Spain", "Germany"),
                          labels = c(1, 2, 3))
dataset$Purchased <- factor(dataset$Purchased,
                            levels = c("No", "Yes"),
                            labels = c(0, 1))