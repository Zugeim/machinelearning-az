# ÁRBOL DE DECISIÓN PARA REGRESIÓN

# Importar el dataset
dataset <- read.csv('Position_Salaries.csv')
dataset <- dataset[, 2:3]

# # Codificar las variables categóricas
# dataset$Level <- factor(dataset$Level,
#                         levels = c("1", "California", "Florida"),
#                         labels = c(1, 2, 3))


# # Dividir los datos en training y testing
# #install.packages("caTools")
# library(caTools)
# set.seed(123)
# split <- sample.split(dataset$Profit, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# testing_set = subset(dataset, split == FALSE)


# Ajustar Árboles de Decisión Regresión al modelo
library(rpart)
regression = rpart(formula= Salary ~ .,
                 data= dataset,
                 control= rpart.control(minsplit= 1))


# Predicción de nuevos resulados con Árbol de Decisión de Regresión 
y_pred = predict(regression, newdata = data.frame(Level = 6.5))


# Visualización del modelo de Árbol de Decisión de regresión
library(ggplot2)
X_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = "red") +
  geom_line(aes(x = X_grid, y = predict(regression, newdata = data.frame(Level = X_grid))),
            color = "blue") +
  ggtitle("Modelo de Árbol de Decisión de Regresión") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")


