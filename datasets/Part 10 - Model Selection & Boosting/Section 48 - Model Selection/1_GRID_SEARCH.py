# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 22:46:36 2023

@author: Imanol
"""

# GRID SEARCH

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importar el data set
dataset = pd.read_csv("Social_Network_Ads.csv")

X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

# Dividir el data set en training y testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) 

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Ajustar el SVM con kernel en el Conjunto de Entrenamiento
from sklearn.svm import SVC
classifier = SVC(kernel= "rbf",
                 random_state= 0)
classifier.fit(X_train, y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Aplicar k-fold cross validation 
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator= classifier, 
                             X= X_train, y= y_train,
                             cv= 10)
accuracies.mean()
accuracies.std()

# Aplicar la mejora de Grid Search para optimizar el modelo y sus parametros ++++++++++++++++++++++++++++++
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']}, # Probamos primero con un kernel lineal
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.001, 0.0001]}] # Probamos despues con un kernel no lineal
grid_search = GridSearchCV(estimator= classifier, 
                           param_grid= parameters,
                           scoring= 'accuracy', # La metrica que usará de medida
                           cv= 10,
                           n_jobs= -1)
grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_
