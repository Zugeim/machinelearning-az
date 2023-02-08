# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:50:23 2023

@author: Imanol
"""

# CLUSTERING JERÁRQUICO


# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importar el data set
dataset = pd.read_csv("Mall_Customers.csv")

X = dataset.iloc[:,[3,4]].values


# Dendograma para averiguar el número óptimo de clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method= "ward"))
plt.title("Dendograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclídea")
plt.show()


# Ajustar el clustering jerárquico a nuestro conjunto de datos
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters= 5,
                affinity= "euclidean",
                linkage= "ward")
y_hc = hc.fit_predict(X)

# Visualización de los clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], c= "red",label= "Cautos")
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], c= "blue",label= "Estandar")
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], c= "green",label= "Objetivo")
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], c= "cyan",label= "Descuidados")
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], c= "magenta",label= "Conservadores")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuación de Gastos (1-100")
plt.legend()
plt.show()
