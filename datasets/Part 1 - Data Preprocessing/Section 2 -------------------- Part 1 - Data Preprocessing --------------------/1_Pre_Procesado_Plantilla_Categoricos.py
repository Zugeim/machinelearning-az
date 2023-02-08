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
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
le_X = preprocessing.LabelEncoder()
X[:, 0] = le_X.fit_transform(X[:, 0])
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
    remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
le_y = preprocessing.LabelEncoder()
y = le_y.fit_transform(y)

