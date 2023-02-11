# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 09:00:50 2023

@author: Imanol
"""

# NATURAL LANGUAGE PROCESSING


# Importación de librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importar dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', 
                      delimiter= '\t',
                      quoting= 3) # Con quoting=3 se ignoran las comillas dobles


# Limpieza de texto
import re
import nltk
nltk.download('stopwords') # Descarga palabras inútiles
from nltk.corpus import stopwords # Carga de palabras inútiles
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, dataset.shape[0]):
    ## 1. Pasar todo a texto
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    
    ## 2. Pasar todo a minuscula
    review = review.lower()
    
    ## 3. Eliminar palabras irrelevantes (preposiciones, conjunciones...)
    ### Separar los datos en un array de strings
    review = review.split()
    ### Eliminar palabras inutiles y quedarse con las raices
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    ## 4. Volver a unir las palabras por espacios
    review = ' '.join(review)
    
    ## 5. Añadir la review al vector de reviews limpias
    corpus.append(review)
    

# Crear el Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# Aplicación de un modelo de clasificación
## Dividir el data set en training y testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

## Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


## Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

## Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
