# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 12:47:00 2023

@author: Imanol
"""

# A PRIORI


# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importar el data set
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header= None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])


# Entrenar el algoritmo a Apriorio
from apyori import apriori # Es una importacion local, esta en la carpeta
rules = apriori(transactions, 
                min_support= 3*7/7500, # Minimo 3 veces comprados al dia
                min_confidence= 0.2, # Muy permisivo saca demasiado, muy restrictivo no saca nada
                min_lift= 3,
                min_legth= 2) # Al menos 2 items en la cesta

# Visualización de los resultados
results = list(rules)

results[1]

