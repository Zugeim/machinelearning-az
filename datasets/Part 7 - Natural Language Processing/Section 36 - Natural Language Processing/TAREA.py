# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:13:25 2023

@author: Imanol
"""

# TAREA NLP


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
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]    
    review = ' '.join(review)
    corpus.append(review)
    

# Crear el Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# Division del data set en training y testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

# PREGUNTA 1

## Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

nb = GaussianNB()
nb.fit(X_train, y_train)

lr = LogisticRegression(random_state= 0)
lr.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors= 5, metric= "minkowski", p= 2) #p = distancia euclidea
knn.fit(X_train, y_train)

svc = SVC(kernel= "linear", random_state= 0)
svc.fit(X_train, y_train)

svc_nl = SVC(kernel= "rbf", random_state= 0)
svc_nl.fit(X_train, y_train)

tc = DecisionTreeClassifier(criterion= "entropy", random_state= 0)
tc.fit(X_train, y_train)

rfc = RandomForestClassifier(n_estimators= 10, criterion= "gini", random_state= 0)
rfc.fit(X_train, y_train)

models_list = [nb, lr, knn, svc, svc_nl, tc, rfc]
models_names =["Naive Bayes", "Logistic Regression",
               "k-nearest neighbors", "SVC", 
               "SVC Non Linear", 'Tree Classifier', 
               "Random Forest Classifier"]

# PREGUNTA 2

## Predicción de los resultados con el Conjunto de Testing
## guardado de accuracy, precisión, recall y F1 score
from sklearn.metrics import confusion_matrix
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
for i, model in enumerate(models_list):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(models_names[i])
    print(cm)
    accuracy = (cm[1][1] + cm[0][0]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])
    precision = cm[1][1] / (cm[1][1] + cm[0][1])
    recall = cm[1][1] / (cm[1][1] + cm[1][0])
    f1_s = 2*precision*recall / (precision + recall)
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1_s)
## Mostrar resultados de las métricas
for i, name in enumerate(models_names):
    print("- Resultados de ", name, " -")
    print("Accuracy: ", round(accuracy_list[i], 2))
    print("Precision: ", round(precision_list[i], 2))
    print("Recall: ", round(recall_list[i], 2))
    print("F1 Score: ", round(f1_list[i], 2))
    print("\n")
    
# PREGUNTA 3
max_e = LogisticRegression(multi_class='multinomial', solver='lbfgs')
max_e.fit(X_train, y_train)
y_pred = max_e.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print('Entropia Máxima')
print(cm)

accuracy = (cm[1][1] + cm[0][0]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])
precision = cm[1][1] / (cm[1][1] + cm[0][1])
recall = cm[1][1] / (cm[1][1] + cm[1][0])
f1_s = 2*precision*recall / (precision + recall)

print("- Resultados de Entropia Máxima -")
print("Accuracy: ", round(accuracy, 2))
print("Precision: ", round(precision, 2))
print("Recall: ", round(recall, 2))
print("F1 Score: ", round(f1_s, 2))
print("\n")

