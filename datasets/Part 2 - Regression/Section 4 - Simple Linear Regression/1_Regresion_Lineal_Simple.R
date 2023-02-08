# Regresion Lineal Simple


# Importar el dataset
dataset <- read.csv('Salary_Data.csv')


# Dividir los datos en training y testing
install.packages("caTools")
library(caTools)
set.seed(123)
split <- sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Ajustar el modelo de regresi?n lineal simple con el conjunto de entrenamiento
regressor = lm(formula = Salary ~ YearsExperience, 
               data = training_set)
summary(regressor)

# Predecir resultados con el conjunto de Test
y_pred = predict(regressor, newdata = testing_set)

# Visualizaci?n de los resultados en el conjunto de entrenamiento
#install.packages("ggplot2")
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), 
             colour = "red") +
  geom_line(aes(x = training_set$YearsExperience, 
                y = predict(regressor, newdata = training_set)),
            colour = "blue") +
  ggtitle("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)") +
  xlab("Años de Experiencia") +
  ylab("Sueldo en ($)")

# Visualización de los resultados en el conjunto de testing
#install.packages("ggplot2")
library(ggplot2)
ggplot() +
  geom_point(aes(x = testing_set$YearsExperience, y = testing_set$Salary), 
             colour = "red") +
  geom_line(aes(x = training_set$YearsExperience, 
                y = predict(regressor, newdata = training_set)),
            colour = "blue") +
  ggtitle("Sueldo vs Años de Experiencia (Conjunto de Testing)") +
  xlab("Años de Experiencia") +
  ylab("Sueldo en ($)")

