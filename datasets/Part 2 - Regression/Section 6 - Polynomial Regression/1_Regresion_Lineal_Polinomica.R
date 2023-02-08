# REGRESIÓN LINEAL POLINÓMICA

# Importar el dataset
dataset <- read.csv('Position_Salaries.csv')
dataset <- dataset[, 2:3]

# # Codificar las variables categ?ricas
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


# Ajustar el modelo de regresión lineal con el Conjunto de Datos
lin_reg = lm(formula = Salary ~ ., 
                data = dataset)
summary(lin_reg)

# Ajustar el modelo de regresión lineal polinomico con el Conjunto de Datos
dataset$Level2 <- dataset$Level^2
dataset$Level3 <- dataset$Level^3
dataset$Level4 <- dataset$Level^4
poly_reg = lm(formula = Salary ~ ., 
                data = dataset)
summary(poly_reg)



# Visualización del modelo lineal
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = "red") +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            color = "blue") +
  ggtitle("Predicción lineal del sueldo en función del nivel del empleado") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")

# Visualización del modelo polinómico
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = "red") +
  geom_line(aes(x = x_grid, y = predict(poly_reg, newdata = data.frame(Level = x_grid,
                                                                       Level2 = x_grid^2,
                                                                       Level3 = x_grid^3,
                                                                       Level4 = x_grid^4))),
            color = "blue") +
  ggtitle("Predicción polinomica del sueldo en función del nivel del empleado") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")

# Predicción de nuevos resultados con Regresión Lineal
y_pred = predict(lin_reg, newdata = data.frame(Level = 6.5))

# Predicción de nuevos resulados con Regresión Polinómica
y_pred = predict(poly_reg, newdata = data.frame(Level = 6.5,
                                                Level2 = 6.5^2,
                                                Level3 = 6.5^3,
                                                Level4 = 6.5^4))
