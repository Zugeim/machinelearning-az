# Regresión Lineal Multiple

# Importar el dataset
dataset <- read.csv('50_Startups.csv')

# Codificar las variables categ?ricas
dataset$State <- factor(dataset$State,
                          levels = c("New York", "California", "Florida"),
                          labels = c(1, 2, 3))


# Dividir los datos en training y testing
#install.packages("caTools")
library(caTools)
set.seed(123)
split <- sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Ajustar el modelo de regresión lineal múltiple con el conjunto de entrenamiento
regression = lm(formula = Profit ~ ., 
               data = training_set)
summary(regression)

# Predecir resultados con el conjunto de Testing
y_pred = predict(regression, newdata = testing_set)

# Construir un modelo óptimo con la Eliminación hacia atrás
SL = 0.05
regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, 
               data = training_set)
summary(regression)
# Eliminamos State
regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, 
                data = training_set)
summary(regression)
# Eliminamos Administration
regression = lm(formula = Profit ~ R.D.Spend + Marketing.Spend, 
                data = training_set)
summary(regression)
# Eliminamos Marketing.Spend
regression = lm(formula = Profit ~ R.D.Spend, 
                data = training_set)
summary(regression)

# Automatización 
## Función
backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}
## Aplicación de la función
SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)