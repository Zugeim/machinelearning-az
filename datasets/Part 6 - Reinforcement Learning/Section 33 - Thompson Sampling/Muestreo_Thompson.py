# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:49:58 2023

@author: Imanol
"""

# MUESTREO DE THOMPSON


# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importar el data set
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")


# Algoritmo de Muestreo Thompson
import random
N = 10000
d = 10
number_of_rewards_1 = [0] * d
number_of_rewards_0 = [0] * d
ads_selected = []
total_reward = 0
for n in range(0, N):
    max_random = 0
    ad = 0
    for i in range(0, d):
        random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        number_of_rewards_1[ad] += 1
    else:
        number_of_rewards_0[ad] += 1
    total_reward += reward
        
# Histograma de resultados
plt.hist(ads_selected)
plt.title("Histograma de anuncios")
plt.xlabel("ID del Anuncio")
plt.ylabel("Frecuencia de visualización del anuncio")
plt.show()
