# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 10:51:35 2023

@author: Imanol
"""
# Regresión lineal Multiple

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importar el data set
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Codificar datos categóricos
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [-1])],
    remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Evitar la trampa de las variables ficticias
# Eliminamos una de las columnas Dummy 
X = X[:,1:]

# Dividir el data set en training y testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

# Ajustar el modulo de Regresión lineal múltiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predicción de los resultados en el conjunto de testing
y_pred = regression.predict(X_test)

# Contruir el modelo óptimo de RLM utilizando la Eliminación hacia atrás
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values=X, axis=1)
SL = 0.05
## Creamos una regresión con todas las variables utilizando la clase statmodels que nos devuelve información util
X_opt = X[:, [0, 1, 2, 3, 4, 5]].tolist()
regression_OLS = sm.OLS(endog= y, exog= X_opt).fit()
regression_OLS.summary()
## Eliminamos columna numero 2
X_opt = X[:, [0, 1, 3, 4, 5]].tolist()
regression_OLS = sm.OLS(endog= y, exog= X_opt).fit()
regression_OLS.summary()
## Eliminamos columna numero 1
X_opt = X[:, [0, 3, 4, 5]].tolist()
regression_OLS = sm.OLS(endog= y, exog= X_opt).fit()
regression_OLS.summary()
## Eliminamos columna numero 4
X_opt = X[:, [0, 3, 5]].tolist()
regression_OLS = sm.OLS(endog= y, exog= X_opt).fit()
regression_OLS.summary()
## Eliminamos columna numero 5
X_opt = X[:, [0, 3]].tolist()
regression_OLS = sm.OLS(endog= y, exog= X_opt).fit()
regression_OLS.summary()

# Automatizandolo
""" 
import statsmodels.api as sm
def backwardElimination(x, sl):    
   numVars = len(x[0])    
   for i in range(0, numVars):        
     regressor_OLS = sm.OLS(y, x.tolist()).fit()        
     maxVar = max(regressor_OLS.pvalues).astype(float)        
     if maxVar > sl:            
        for j in range(0, numVars - i):                
          if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
             x = np.delete(x, j, 1)    
   print(regressor_OLS.summary())  
   return x

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
"""

# Automatizandolo en función de p-valor y R cuadrada ajustada
"""
import statsmodels.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])    
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        adjR_before = regressor_OLS.rsquared_adj.astype(float)        
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]                    
                    x = np.delete(x, j, 1)                    
                    tmp_regressor = sm.OLS(y, x.tolist()).fit()                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))                        
                        x_rollback = np.delete(x_rollback, j, 1)    
                        print (regressor_OLS.summary())                        
                        return x_rollback 
                    else:
                        continue
    regressor_OLS.summary()
    return x            
        

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
"""