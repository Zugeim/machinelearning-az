# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 22:19:02 2023

@author: Imanol
"""

# K - FOLD CROSS VALIDATION


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

# Aplicar k-fold cross validation +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator= classifier, 
                             X= X_train, y= y_train,
                             cv= 10)
accuracies.mean()
accuracies.std()