# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 14:48:56 2023

@author: Imanol
"""

# Plantilla de Pre Procesado de Datos Faltantes

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

