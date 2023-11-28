# Limpa o ambiente de trabalho
rm(list=ls())

#Bibliotecas
library(caret)

#Importa CSVs
treino <- read.csv("treino.csv")

#Transforma colunas da TARGET de texto para numeros
treino$y <- factor(treino$y, levels = c('teens', 'twenties', 'thirties', 'fourties', 'fifties'))
treino$y_num <- as.numeric(treino$y) - 1
treino <- treino[, -which(names(treino) == 'y')]

#--------------------Implementa Pre-Processamento - PCA
#Diminui a dimensionalidade e Verifica quais features sao mais discriminativas
#trans <- preProcess(treino_normalizado, method = c("pca"))
#PC <- predict(trans, treino_normalizado)
#-----------------------------------------------

#--------------------Inclui TARGET pós Pre-Processamento
# Incluir as colunas da variável de destino nas duas primeiras componentes principais
#PC_y <- cbind(PC[, 1:2], y_num = as.numeric(treino$y_num))
# Plotar um gráfico de dispersão
#plot(PC_y$PC1, PC_y$PC2, pch = 16, col = PC_y$y_num,
#     main = "Scatter Plot das PC1 e PC2 com a variável de destino", xlab = "PC1", ylab = "PC2")
#-----------------------------------------------

#--------------------Gera gráfico de Autovalores
# Calcula os autovalores
#eigenvalues <- eigen(cov(PC))$values
# Calcula a proporção da variância explicada
#variance_explained <- eigenvalues / sum(eigenvalues)
# Gera o gráfico dos autovalores
#plot(variance_explained, type = "b", ylab = "Proporção da Variância Explicada", xlab = "Componente Principal", main = "Gráfico dos Autovalores")
#-----------------------------------------------

# Seleciona as 10 primeiras componentes principais
#treino <- PC[, (1:10)]

#-----------------------------------------------
#-----------------------------------------------
#------------------------ KDE ------------------
#-----------------------------------------------
#-----------------------------------------------

#--------------------Funções
# define a função KDE
pdfKDE <- function(xi, N, x) {
  sum <- 0
  h <- 1.06 * sd(as.matrix(x)) * N^(-1/5)
  
  for (i in 1:N) {
    xi_matrix <- as.matrix(xi)
    x_i_matrix <- as.matrix(x[i, ])
    
    exp_values <- exp(-(1/(2*h^2) * (t(x_i_matrix - xi_matrix) %*% (x_i_matrix - xi_matrix))))
    
    # Substitui infinitos ou NaNs por 0
    exp_values[!is.finite(exp_values)] <- 0
    
    sum <- sum + sum(exp_values)
  }
  
  p <- (1/(N * (sqrt(2 * pi * h))^N)) * sum
  return(p)
}

# Validação cruzada com 10 folds usando bayes
bayes_classifier <- function(x_train, y_train, x_test){
  class_probabilities <- table(y_train) / length(y_train)
  
  y_hat <- numeric(length = nrow(x_test))
  
  for (i in 1:nrow(x_test)) {
    class_scores <- numeric(length = max(y_train))
    
    for (c in levels(factor(y_train))) {
      class_indices <- which(y_train == c)
      p <- pdfKDE(x_test[i, ], N = length(class_indices), x = x_train[class_indices, ])
      
      class_scores[as.numeric(c)] <- sum(log(p) * class_probabilities[c])
    }
    
    predicted_class <- which.max(class_scores)
    y_hat[i] <- predicted_class
  }
  
  return(y_hat)
}

#-----------------------------------------------
# Separar em X e Y
x <- treino[, 2:40]  # Considerando as 39 colunas de características
y <- treino[, "y_num"]
#x <- as.matrix(treino[, !names(treino) %in% c('y_num', 'id')])
#y <- as.vector(treino$y_num)  # Ajustado para vetor
index <- sample(1:nrow(x), length(1:nrow(x)))

# Obtem-se acurácia obtida para cada iteração, o desvio padrão das acurácias e a média das acurácias
accuracy <- matrix(NA, nrow = 10, ncol = 1)
j <- 1
best <- 0
# Loop de validação cruzada
for (i in seq(20, 200, 20)) {
  test <- index[(i - 19):i]
  train <- index[-index[(i - 19):i]]
  
  x_train <- x[train, ]
  y_train <- y[train]
  x_test <- x[test, ]
  y_test <- y[test]
  
  # Chama o classificador bayesiano
  y_hat <- bayes_classifier(x_train, as.numeric(y_train), x_test)
  
  # Calcula a acurácia
  aux <- sum(y_test == y_hat) / length(y_test)
  accuracy[j, ] <- aux
  if(aux > best){
    best_train <- train
    best_test <- test
    save_index <- j
    best <- aux 
  }
  j <- j+1
}

print(accuracy)
print(apply(accuracy, 2, sd))
print(apply(accuracy, 2, mean))