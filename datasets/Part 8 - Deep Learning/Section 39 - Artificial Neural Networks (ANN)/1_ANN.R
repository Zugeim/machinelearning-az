# REDES NEURONALES ARTIFICIALES (RNN)


# Importar el dataset
dataset <- read.csv('Churn_Modelling.csv')
dataset <- dataset[, 4:14]

# Codificar los factores para la red neuronal artificial
dataset$Geography <- as.numeric(factor(dataset$Geography,
                                       levels = c("France", "Spain", "Germany"),
                                       labels = c(1, 2, 3)))
dataset$Gender <- as.numeric(factor(dataset$Gender,
                                    levels = c("Female", "Male"),
                                    labels = c(1, 2)))


# Dividir los datos en training y testing
#install.packages("caTools")
library(caTools)
set.seed(123)
split <- sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Escalado de variables
training_set[,-11] = scale(training_set[,-11])
testing_set[,-11] = scale(testing_set[,-11])


# Crear la red neuronal
#install.packages("h2o")
library(h2o)
## Hay que conectarse a la instancia de h2o
h2o.init(nthreads = -1) # Número de hilos para hacer el calculo (Todos los nucleos menos 1)
classifier = h2o.deeplearning(y = "Exited", # Variable dependiente
                              training_frame = as.h2o(training_set), # Variables predictoras
                              activation = "Rectifier", # Función de activación
                              hidden = c(6, 6), # Número de capas ocultas y número de neuronas (2 capas y usar promedio de capas de entrada y salida para el número de neuronas)
                              epochs = 100, # Número de iteraciones
                              train_samples_per_iteration = -2) # Número de muestras por iteración

# Predicción de los resulados con el Conjunto de Testing
prob_pred = h2o.predict(classifier, newdata = as.h2o(testing_set[,-11]))

y_pred = (prob_pred > 0.5)
y_pred = as.vector(y_pred)

# Crear la matriz de confusión
cm = table(testing_set[,11], y_pred)

# Cerrar la sesión de H2O
h2o.shutdown()

