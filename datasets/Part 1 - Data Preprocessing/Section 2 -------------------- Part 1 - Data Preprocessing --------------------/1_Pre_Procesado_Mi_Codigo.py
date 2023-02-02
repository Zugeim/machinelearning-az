# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 12:04:08 2023

@author: Imanol
"""


# Plantilla de Pre Procesado

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values

# Tratamiento de los NAs
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:,1:3])

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_X.fit_transform(y)

# Dividir el data set en training y testing
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)