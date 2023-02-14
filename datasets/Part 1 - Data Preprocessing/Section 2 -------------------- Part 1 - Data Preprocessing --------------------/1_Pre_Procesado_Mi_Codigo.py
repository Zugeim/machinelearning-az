# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 12:04:08 2023

@author: Imanol
"""


# Plantilla de Pre Procesado

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv("Data.csv")




X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values


# Tratamiento de los NAs
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy="mean")
imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:,1:3])

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



from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

#Categóricas
cat = dataset.select_dtypes('O')

## Instanciamos
ohe = OneHotEncoder(sparse= False)

## Entrenamos y Aplicamos
cat_ohe = ohe.transform(cat)

#Ponemos los nombres
cat_ohe = pd.DataFrame(cat_ohe, columns = ohe.get_feature_names_out(input_features = cat.columns)).reset_index(drop = True)

## Eliminamos las columnas necesarias para que no exista
## colinealidades
cat_ohe.drop(columns= ['Country_France', 'Purchased_No'], 
            inplace= True) 

## Cambiamos el nombre de la variable dependiente
cat_ohe.rename(columns={"Purchased_Yes": "Purchased"},
               inplace= True)

# Dividir el data set en training y testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Escalado de variables
from sklearn.preprocessing import StandardScaler
num = dataset.select_dtypes('number').reset_index(drop = True)
sc = StandardScaler()
num_sc = sc.fit_transform(num)

num = pd.DataFrame(num_sc, 
                  columns sc.get_feature_names_out(
 	input_features = num.columns)).reset_index(drop = True)


