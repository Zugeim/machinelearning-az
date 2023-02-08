# CLUSTERING JERARQUICO

# Importar el data set
dataset = read.csv("Mall_Customers.csv")

X = dataset[,c(4,5)]


# Dendograma para averiguar el número óptimo de clusters
dendogram = hclust(dist(X, method = "euclidean"),
                   method = "ward.D")
plot(dendogram,
      main = "Dendograma",
      xlab = "Clientes del centro comercial",
      ylab = "Distancia Euclidea")


# Ajustar el clustering jerárquico a nuestro conjunto de datos
hc = hclust(dist(X, method = "euclidean"),
                   method = "ward.D")
y_hc = cutree(hc, k = 5)


# Visualización de los clusters
library(cluster)
clusplot(X,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = "Custering de Clientes",
         xlab = "Ingresos anuales",
         ylab = "Puntuación (1-100)")