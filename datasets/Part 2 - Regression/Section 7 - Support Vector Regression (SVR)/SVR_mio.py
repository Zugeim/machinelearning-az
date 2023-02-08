# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 10:30:24 2023

@author: Imanol
"""

# SVR

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importar el data set
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values


# Dividir el data set en training y testing
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 
"""

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))


# Ajustar la regresión con el dataset
from sklearn.svm import SVR
regression = SVR(kernel= "rbf")
regression.fit(X, y)


# Predicción de nuestros modelos
y_pred = regression.predict(sc_X.transform(np.array([[6.5]]))).reshape(1,-1)
y_pred = sc_y.inverse_transform(y_pred)

# Cambios de escala para visualizar
X = sc_X.inverse_transform(X)
y = sc_y.inverse_transform(y)


# Visualización de los resultados del Modelo PolinómicoSVR
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color= "red")
plt.plot(X_grid, 
         sc_y.inverse_transform(regression.predict(sc_X.transform(X_grid)).reshape(-1,1)), 
         color= "blue")
plt.title("Modelo de Regresión")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()