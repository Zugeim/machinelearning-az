# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 15:11:26 2023

@author: Imanol
"""

# REDES NEURONALES ARTIFICIALES (ANN)


# Instalar Theano
# Esta parte no la he conseguido
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Instalar Tensorflow y Keras
# conda install -c conda-forge keras

# Parte 1 - Pre procesado de datos


# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv("Churn_Modelling.csv")

X = dataset.iloc[:,3:13].values
y = dataset.iloc[:, 13].values

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   
    remainder='passthrough'                        
)
X = onehotencoder.fit_transform(X)
X = X[:, 1:] # Elminamos el problema de colinealidad


# Dividir el data set en training y testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Parte 2 - Construir la RNA ----------------------------------------------------------------------------------

# Importar Keras y librerías adicionales
import keras
from keras.models import Sequential
from keras.layers import Dense

# Inicializar la RNA
classifier = Sequential()

# Añadir las capas de entrada y primera capa oculta
classifier.add(Dense(units= 6, # Nº nodos de capa oculta (escoger la media de las entradas y salidas de la red)
                     kernel_initializer= 'uniform', # Distribución de los pesos en la inicialización
                     activation= 'relu', # Función de activación (Rectificador lineal unitario)
                     input_dim= 11)) # Especificar nº de nodos de entrada

# Añadir la segunda capa oculta
## input_dim ya no hace falta porque ya sabe que viene de una capa de 6 nodos          
classifier.add(Dense(units= 6, kernel_initializer= 'uniform', activation= 'relu')) 

# Añadir la capa de salida
classifier.add(Dense(units= 1, kernel_initializer= 'uniform', activation= 'sigmoid')) 

# Compilar la RNA
classifier.compile(optimizer= 'adam', # Busca las soluciones optimas (optimizador de adam))
                   loss= 'binary_crossentropy',# Función de perdidas
                   metrics= ['accuracy'])# Utiliza la precision como medida de la bondad del modelo

# Ajustamos la RNA al Conjunto de entrenamiento
classifier.fit(X_train, y_train,
               batch_size= 10, # Número de elementos que pasan antes de corregirse
               epochs=100) # Número de iteraciones sobre el conjunto total


# Parte 3 - Evaluar el modelo y calcular predicciones finales--------------------------------------------------

# Predicción de los resultados con el Conjunto de Testing
prob_pred = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
y_pred = (prob_pred > 0.5) # Como son probabilidades escogemos el umbral
cm = confusion_matrix(y_test, y_pred)
