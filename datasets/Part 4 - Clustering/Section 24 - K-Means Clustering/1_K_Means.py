# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 08:46:15 2023

@author: Imanol
"""

# K - MEANS

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importar el data set
dataset = pd.read_csv("Mall_Customers.csv")

X = dataset.iloc[:,[3,4]].values


# Método del codo para averiguar el número óptimo de clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters= i,
                    init= "k-means++",
                    max_iter= 300,
                    n_init= 10,
                    random_state= 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title("Método del codo")
plt.xlabel("k (Número de clusters)")
plt.ylabel("WCSS (k)")
plt.show()


# Aplicar el método k-means para segmentar el data set
kmeans = KMeans(n_clusters= 5,
                init= "k-means++",
                max_iter= 300,
                n_init= 10,
                random_state= 0)
y_kmeans = kmeans.fit_predict(X)


# Visualización de los clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], c= "red",label= "Cautos")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], c= "blue",label= "Estandar")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], c= "green",label= "Objetivo")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], c= "cyan",label= "Descuidados")
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], c= "magenta",label= "Conservadores")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],c= "yellow",label= "Baricentros")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuación de Gastos (1-100")
plt.legend()
plt.show()
