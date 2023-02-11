# NATURAL LANGUAGE PROCESSING


# Importar el dataset
dataset_original <- read.csv('Restaurant_Reviews.tsv',
                    quote = '', # Cualquier cosa puede ser un texto
                    stringsAsFactors = FALSE, # Para que no transforme los textos a factor
                    sep = '\t')


# Limpieza de texto
#install.packages("tm")
#install.packages("SnowballC")
library(tm)
library(SnowballC)

corpus = VCorpus(VectorSource(dataset_original$Review))

## 1. Pasar todo a minuscula
corpus = tm_map(corpus, content_transformer(tolower))

## 2. Eliminar lo que no sea texto
### Eliminar números
corpus = tm_map(corpus, removeNumbers)
### Eliminar signos de puntuación
corpus = tm_map(corpus, removePunctuation)

## 3. Eliminar palabras irrelevantes (preposiciones, conjunciones...)
corpus = tm_map(corpus, removeWords, stopwords(kind = 'en'))

## 4. Obtener raices de las palabras
corpus = tm_map(corpus, stemDocument)

## 5. Eliminar espacios adicionales
corpus = tm_map(corpus, stripWhitespace)


# Crear el modelo Bag of Words
dtm = DocumentTermMatrix(corpus)

## Filtrar las palabras más frecuentes
dtm = removeSparseTerms(dtm, 0.999) # 99.9% de palabras más frecuentes


# Aplicar un modelo

dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Codificar las variables categóricas
dataset$Liked <- factor(dataset$Liked,
                            levels = c(0, 1))

# Dividir los datos en training y testing
#install.packages("caTools")
library(caTools)
set.seed(123)
split <- sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Ajustar el clasificador en el Conjunto de Entrenamiento
library("randomForest")
classifier = randomForest(x = training_set[,-ncol(dataset)],
                          y = training_set$Liked,
                          ntree = 10)

# Predicción de los resulados con el Conjunto de Testing
y_pred = predict(classifier, newdata = testing_set[,-ncol(dataset)])

# Crear la matriz de confusión
cm = table(testing_set[,ncol(dataset)], y_pred)

# Precisión
(cm[1] + cm[4]) / 200
