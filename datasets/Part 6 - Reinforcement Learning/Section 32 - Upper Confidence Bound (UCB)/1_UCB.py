# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 08:55:25 2023

@author: Imanol
"""

# UPPER CONFIDENCE BOUND (UCB)

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Cargar el data set
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")


# Algoritmo UCB
import math
N = 10000
d = 10
number_of_selections = [0] * d
sums_of_reward = [0] * d
ads_selected = []
total_reward = 0
for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range(0, d):
        if(number_of_selections[i] > 0):
            average_reward = sums_of_reward[i] / number_of_selections[i]
            delta_i = math.sqrt(3/2*math.log(n+1)/number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_reward[ad] += reward
    total_reward += reward
        
# Histograma de resultados
plt.hist(ads_selected)
plt.title("Histograma de anuncios")
plt.xlabel("ID del Anuncio")
plt.ylabel("Frecuencia de visualización del anuncio")
plt.show()
        
        