# K - MEANS

# Importar el data set
dataset = read.csv("Mall_Customers.csv")

X = dataset[,c(4,5)]


# Método del codo para averiguar el número óptimo de clusters
set.seed(6)
wcss = vector()
for (i in 1:10) {
  wcss[i] <- sum(kmeans(X, i)$withinss)
}
plot(1:10, wcss, type = 'b', main = "Método del codo",
     xlab = "Número de clusters (k)", ylab = "WCSS(k)")



# Aplicar el método k-means para segmentar el data set
set.seed(29)
kmeans <- kmeans(X, 5, iter.max = 300, nstart = 10)


# Visualización de los clusters
library(cluster)
clusplot(X,
         kmeans$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = "Custering de Clientes",
         xlab = "Ingresos anuales",
         ylab = "Puntuación (1-100)")