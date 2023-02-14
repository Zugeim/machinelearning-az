# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 08:24:04 2023

@author: Imanol
"""

# REDES NEURONALES CONVOLUCIONALES (CNN)

# Instalar Theano
# Esta parte no la he conseguido
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Instalar Tensorflow y Keras
# conda install -c conda-forge keras




# Parte 1 - Construir el modelo de CNN ---------------------------------


# Importar las librerías y paquetes
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

# Inicializar la CNN
classifier = Sequential()

# Paso 1 - Convolución
classifier.add(Convolution2D(filters= 32, kernel_size= (3, 3), # 32 Mapas de características de 3x3
                             input_shape= (64, 64, 3), # Tamaño de las imagenes y el 3 es el color
                             activation= 'relu' )) # Función de activación

# Paso 2 - Max Pooling
classifier.add(MaxPool2D(pool_size= (2,2))) # Tamaño del max pooling

# Paso 3 - Flattering (aplanado)
classifier.add(Flatten())

# Paso 4 - Full Connection
classifier.add(Dense(units= 128, # Se elige un número de nodos
                     activation= "relu")) # Función de activación
classifier.add(Dense(units= 1, activation= "sigmoid"))  # Capa de salida                

# Compilar la CNN
classifier.compile(optimizer= "adam", loss= "binary_crossentropy", metrics= ["accuracy"])




# Parte 2 - Ajustar la CNN a las imágenes para entrenar ----------------------------------------
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255, # Transforma los pixels de 0-255 a 0-1
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_dataset = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size=(64, 64), # Muestrea a 64x64 
                                                     batch_size=32, # Lotes antes de las correcciones
                                                     class_mode='binary') # Método de clasificación

testing_dataset = test_datagen.flow_from_directory('dataset/test_set',
                                                   target_size=(64, 64),
                                                   batch_size=32,
                                                   class_mode='binary')

#steps_per_epoch = len(training_dataset) // batch_size 
#validation_steps = len(testing_dataset) // batch_size

classifier.fit(training_dataset,
               steps_per_epoch=250, # Con cuantas imagenes entrenará en cada epoch
               epochs=25,
               validation_data=testing_dataset,
               validation_steps=2000)