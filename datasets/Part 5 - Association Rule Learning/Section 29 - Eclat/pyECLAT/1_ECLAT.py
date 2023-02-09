# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:22:19 2023

@author: Imanol
"""

# ECLAT


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
from pyECLAT import ECLAT # Es una importacion local, esta en la carpeta
eclat_instance = ECLAT(data = transactions, verbose= True) 

get_ECLAT_indexes, get_ECLAT_supports = eclat_instance.fit(min_support= 0.003,
                                                           min_combination= 1,
                                                           separator=,
                                                           verbose= True)
# Visualización de los resultados
results = list(rules)

results[1]