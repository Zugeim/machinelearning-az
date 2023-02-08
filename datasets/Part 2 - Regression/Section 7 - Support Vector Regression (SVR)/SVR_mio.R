# SVR

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


# Ajustar SVR al modelo
#install.packages("e1071")
library(e1071)
regression = svm(formula= Salary ~ .,
                 data = dataset,
                 type= "eps-regression",
                 kernel= "radial")


# Predicción de nuevos resulados con Regresión 
y_pred = predict(regression, newdata = data.frame(Level = 6.5))


# Visualización del modelo de regresión
library(ggplot2)
#x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = "red") +
  geom_line(aes(x = dataset$Level, y = predict(regression, newdata = data.frame(Level = dataset$Level))),
            color = "blue") +
  ggtitle("Modelo de Regresión") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")


