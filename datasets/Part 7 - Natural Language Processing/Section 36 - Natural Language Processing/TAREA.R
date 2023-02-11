# TAREA SIN ACABAR


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
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords(kind = 'en'))
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)


# Crear el modelo Bag of Words
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999) # 99.9% de palabras más frecuentes


dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked
dataset$Liked <- factor(dataset$Liked,
                        levels = c(0, 1))

# Dividir los datos en training y testing
#install.packages("caTools")
library(caTools)
set.seed(123)
split <- sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# PREGUNTA 1

# Ajustar los clasificadores en el Conjunto de Entrenamiento
library("randomForest")
library("class")
library("e1071")
library("rpart")

nb = naiveBayes(x = training_set[,-ncol(dataset)],
                y = training_set$Liked)
y_pred_nb = predict(nb, newdata = testing_set[,-ncol(dataset)])

lr = glm(formula = Liked ~ .,
         data = training_set,
         family = binomial)
y_pred_lr = predict(lr, newdata = testing_set[,-ncol(dataset)])

y_pred_knn = knn(train = training_set[,-ncol(dataset)],
          test = testing_set[,-ncol(dataset)],
          cl = training_set$Liked,
          k = 5)

svc = svm(formula = Liked ~ .,
          data = training_set,
          type = "C-classification",
          kernel = "linear")
y_pred_svc = predict(svc, newdata = testing_set[,-ncol(dataset)])

svc_nl = svm(formula = Liked ~ .,
              data = training_set,
              type = "C-classification",
              kernel = "radial")
y_pred_svc_nl = predict(svc_nl, newdata = testing_set[,-ncol(dataset)])

tc = rpart(formula = Liked ~ .,
           data = training_set)
y_pred_tc = predict(tc, newdata = testing_set[,-ncol(dataset)])

rfc = randomForest(x = training_set[,-ncol(dataset)],
                   y = training_set$Liked,
                   ntree = 10)
y_pred_rfc = predict(rfc, newdata = testing_set[,-ncol(dataset)])

models_names = c("Naive Bayes", "Logistic Regression",
                 "k-nearest neighbors", "SVC", 
                 "SVC Non Linear", 'Tree Classifier', 
                 "Random Forest Classifier")
y_pred = c(y_pred_nb, y_pred_lr, y_pred_knn, y_pred_svc, y_pred_svc_nl, y_pred_tc, y_pred_rfc)


# PREGUNTA 2

# Predicción de los resulados con el Conjunto de Testing
accuracy_list = numeric()
precision_list = numeric()
recall_list = numeric()
f1_list = numeric()
metricas = data.frame("Naive Bayes" = numeric(4), "Logistic Regression" = numeric(4),
                      "k-nearest neighbors" = numeric(4), "SVC" = numeric(4), 
                      "SVC Non Linear" = numeric(4), 'Tree Classifier' = numeric(4), 
                      "Random Forest Classifier" = numeric(4));
for (i in seq_along(models_names)) {
  
  cm = table(testing_set[,ncol(dataset)], y_pred)
  print(models_names[i])
  print(cm)
  
  accuracy = (cm[1][1] + cm[0][0]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])
  precision = cm[1][1] / (cm[1][1] + cm[0][1])
  recall = cm[1][1] / (cm[1][1] + cm[1][0])
  f1_s = 2*precision*recall / (precision + recall)
  
  
  
  
}

y_pred = predict(models_list[7], newdata = testing_set[,-ncol(dataset)])
models_names[7]
