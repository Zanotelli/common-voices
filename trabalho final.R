# Limpa o ambiente de trabalho
rm(list=ls())

#Bibliotecas
library(caret)

# Importa CSVs
treino <- read.csv("treino.csv")

# Transforma colunas da TARGET de texto para numeros
treino$y <- factor(treino$y, levels = c('teens', 'twenties', 'thirties', 'fourties', 'fifties'))
treino$y_num <- as.numeric(treino$y) - 1
treino <- treino[, -which(names(treino) == 'y')]
y <- treino$y_num

#--------------------Implementa Pre-Processamento - PCA
# Diminui a dimensionalidade e Verifica quais features sao mais discriminativas
# trans <- preProcess(treino, method = c("pca"))
# PC <- predict(trans, treino)
#-----------------------------------------------

#--------------------Inclui TARGET pós Pre-Processamento
# Incluir as colunas da variável de destino nas duas primeiras componentes principais
# PC_y <- cbind(PC[, 1:2], y_num = as.numeric(treino$y_num))
# Plotar um gráfico de dispersão
# plot(PC_y$PC1, PC_y$PC2, pch = 16, col = PC_y$y_num,
#     main = "Scatter Plot das PC1 e PC2 com a variável de destino", xlab = "PC1", ylab = "PC2")
#-----------------------------------------------

#--------------------Gera gráfico de Autovalores
# Calcula os autovalores
# eigenvalues <- eigen(cov(PC))$values

# Calcula a proporção da variância explicada
# variance_explained <- eigenvalues / sum(eigenvalues)
# Gera o gráfico dos autovalores
# plot(variance_explained, type = "b", ylab = "Proporção da Variância Explicada", xlab = "Componente Principal", main = "Gráfico dos Autovalores")
#-----------------------------------------------

# Seleciona as 10 primeiras componentes principais
# treino <- PC[, (1:10)]

#-----------------------------------------------
#-----------------------------------------------
#------------------------ KDE ------------------
#-----------------------------------------------
#-----------------------------------------------

#--------------------Funções--------------------
# Define a função KDE
pdfKDE <- function(xi, N, x, h) {
  sum <- 0
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

# Função do classificador bayesiano ajustada para aceitar h como argumento
bayes_classifier <- function(x_train, y_train, x_test, h){
  class_probabilities <- table(y_train) / length(y_train)
  
  y_hat <- numeric(length = nrow(x_test))
  
  for (i in 1:nrow(x_test)) {
    class_scores <- numeric(length = max(y_train))
    
    for (c in levels(factor(y_train))) {
      class_indices <- which(y_train == c)
      p <- pdfKDE(x_test[i, ], N = length(class_indices), x = x_train[class_indices, ], h = h)
      
      class_scores[as.numeric(c)] <- sum(log(p) * class_probabilities[c])
    }
    
    predicted_class <- which.max(class_scores)
    y_hat[i] <- predicted_class
  }
  
  return(y_hat)
}

#-------------------TEN FOLD-----------------------
# Separar em X e Y
x <- treino[, 2:40]  
y <- (as.matrix(as.numeric(treino[, "y_num"])))
set.seed(123)  # Define a semente para reproducibilidade
index <- sample(1:nrow(x), 0.8 * nrow(x))  # 80% para treino
#index <- sample(1:nrow(x), length(1:nrow(x)))

# Loop de validação cruzada com pesquisa aleatória para o parâmetro h
set.seed(123)  # Define a semente para reproducibilidade
accuracy <- matrix(NA, nrow = 10, ncol = 1)
best <- 0

for (j in seq_len(10)) {
  n_te <- 100 #dim(treino)[1]*0.1
  conjunto_teste <- ((j - 1) * n_te + 1):(j * n_te)
  test <- index[conjunto_teste]
  train <- index[-index[conjunto_teste]]
  
  x_train <- treino[train, 2:40]
  y_train <- treino[train, "y_num"]
  x_test <- treino[test, 2:40]
  y_test <- treino[test, "y_num"]
  
  # Pesquisa aleatória para o parâmetro h
  h_values <- seq(0.1, 0.5, by = 0.1)
  best_accuracy <- 0
  best_h <- 0
  
  for (h in h_values) {
    # Chama o classificador bayesiano com o valor atual de h
    y_hat <- bayes_classifier(x_train, y_train, x_test, h)
    
    # Calcula a acurácia
    aux <- sum(y_test == y_hat) / length(y_test)
    
    # Atualiza o melhor valor de h e acurácia se necessário
    if (aux > best_accuracy) {
      best_train <- train
      best_test <- test
      best_h <- h
      best_accuracy <- aux
    }
  }
  
  # Exibe o melhor valor de h encontrado para cada fold
  cat("Melhor valor de h encontrado para fold", j, ":", best_h, "\n")
  cat("Acurácia: ", round(best_accuracy, 4), "\n")
  # Armazena a acurácia para este fold
  accuracy[j, ] <- best_accuracy
}

# Exibe as acurácias e estatísticas
print("Acurácias:")
print(accuracy)
cat("Média da Acurácia:", mean(accuracy), "\n")
cat("Desvio Padrão da Acurácia:", sd(accuracy), "\n")

#-------------------------------------------------------

#treina os dados 
test <- best_test
train <- best_train
x_train <- x[train, ]
y_train <- y[train]
x_test <- x[test, ]
y_test <- y[test]

y_hat <- integer(length(y_test))
espaco_de_verossimilhanca <- matrix(0, nrow = length(y_test), ncol = 2)  # Inicializa com zeros

# Calcula as probabilidades para cada classe
p_classes <- numeric(5)
for (class_label in 0:4) {
  p_classes[class_label + 1] <- sum(y_train == class_label) / length(y_train)
}
c1 = x_train[y_train == 0, ]
c2 = x_train[y_train == 1, ]
y_hat <- integer(length(y_test))
h=0.05

for (i in 1:length(y_test)) {
  # Cálculo das densidades para cada classe
  p_values <- numeric(5)
  for (class_label in 0:4) {
    p_values[class_label + 1] <- pdfKDE(x_test[i, ], nrow(x_train[y_train == class_label, ]), x_train[y_train == class_label, ], best_h)
  }
  
  # Cálculo da razão de probabilidades para cada par de classes
  K_values <- numeric(5)
  for (class_label in 0:4) {
    K_values[class_label + 1] <- (p_values[class_label + 1] * p_classes[class_label + 1]) / sum(p_values * p_classes)
  }
  
  # Atribuição da previsão com base na classe com maior K
  y_hat[i] <- which.max(K_values) - 1
  
  # Preenche a matriz espaco_de_verossimilhanca com os valores das densidades
  espaco_de_verossimilhanca[i, ] <- c(p_values[1], p_values[2])  # Ajuste conforme apropriado
}

#------------------------------------------------------

# Avaliação do desempenho
final_accuracy <- sum(y_test == y_hat) / length(y_test)
print(paste("Acurácia Final: ", final_accuracy)) 
