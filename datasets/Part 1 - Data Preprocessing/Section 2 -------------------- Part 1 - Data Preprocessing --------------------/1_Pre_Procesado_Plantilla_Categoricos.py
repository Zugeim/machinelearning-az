# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 14:47:49 2023

@author: Imanol
"""

# Plantilla de Pre Procesado de Datos Categóricos

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_X.fit_transform(y)

