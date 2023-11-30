# Limpa o ambiente de trabalho
rm(list=ls())

#Bibliotecas
library(caret)
library(MASS)
install.packages("e1071")
library(e1071)

# Importa CSVs
treino <- read.csv("treino.csv")

# Transforma colunas da TARGET de texto para numeros
treino$y <- factor(treino$y, levels = c('teens', 'twenties', 'thirties', 'fourties', 'fifties'))
treino$y_num <- as.numeric(treino$y) - 1
treino <- treino[, -which(names(treino) == 'y')]
y <- treino$y_num

#--------------------Implementa Pre-Processamento - PCA
# Diminui a dimensionalidade e Verifica quais features sao mais discriminativas
trans <- preProcess(treino, method = c("pca"))
PC <- predict(trans, treino)
#-----------------------------------------------

#--------------------Inclui TARGET pós Pre-Processamento
# Incluir as colunas da variável de destino nas duas primeiras componentes principais
PC_y <- cbind(PC[, 1:2], y_num = as.numeric(treino$y_num))
# Plotar um gráfico de dispersão
plot(PC_y$PC1, PC_y$PC2, pch = 16, col = PC_y$y_num,
     main = "Scatter Plot das PC1 e PC2 com a variável de destino", xlab = "PC1", ylab = "PC2")
#-----------------------------------------------

#--------------------Gera gráfico de Autovalores
# Calcula os autovalores
eigenvalues <- eigen(cov(PC))$values

# Calcula a proporção da variância explicada
variance_explained <- eigenvalues / sum(eigenvalues)
# Gera o gráfico dos autovalores
plot(variance_explained, type = "b", ylab = "Proporção da Variância Explicada", xlab = "Componente Principal", main = "Gráfico dos Autovalores")
#-----------------------------------------------

# Seleciona as 10 primeiras componentes principais
treino <- PC[, (1:7)]

#-----------------------------------------------
#-----------------------------------------------
#------------------------ KDE ------------------
#-----------------------------------------------
#-----------------------------------------------

#--------------------Funções--------------------
pdfKDEOld <- function(xi, N, x, h) {
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
bayes_classifier <- function(x_train, y_train, x_test, h) {
  class_probabilities <- table(y_train) / length(y_train)
  num_classes <- length(levels(factor(y_train)))
  
  y_hat <- numeric(length = nrow(x_test))
  
  for (i in 1:nrow(x_test)) {
    class_scores <- numeric(length = num_classes)
    
    for (c in 1:num_classes) {
      class_indices <- which(y_train == levels(factor(y_train))[c])
      #p <- pdfKDE(x_test[i,], length(class_indices), x_train[class_indices,])
      
      class_scores[c] <- sum(log(p$y) + log(class_probabilities[c]))
    }
    
    predicted_class <- which.max(class_scores)
    y_hat[i] <- levels(factor(y_train))[predicted_class]
  }
  
  return(y_hat)
}


#-------------------TEN FOLD-----------------------

set.seed(123)  # Define a semente para reproducibilidade
index <- sample(1:nrow(treino), 0.8 * nrow(treino))  # 80% para treino
#index <- sample(1:nrow(x), length(1:nrow(x)))

# Loop de validação cruzada com pesquisa aleatória para o parâmetro h
set.seed(123)  # Define a semente para reproducibilidade
h_values <- seq(0.1, 1, by = 0.05)
accuracy <- matrix(NA, nrow = 10, ncol = 1)
best <- 0

for (j in seq_len(10)) {
  n_te <- 100 #dim(treino)[1]*0.1
  conjunto_teste <- ((j - 1) * n_te + 1):(j * n_te)
  test <- index[conjunto_teste]
  train <- index[-index[conjunto_teste]]
  
  x_train <- treino[train,]
  y_train <- y[train]
  x_test <- treino[test,]
  y_test <- y[test]

  
  svm_model <- svm(x_train, y_train, kernel = "radial", cost = 10)
  y_svm <- predict(svm_model, x_test)
  accuracy[j] <- sum(y_test == round(y_svm)) / length(y_test)
  
}

# Exibe as acurácias e estatísticas
print("Acurácias:")
print(accuracy)
cat("Média da Acurácia:", mean(accuracy), "\n")
cat("Desvio Padrão da Acurácia:", sd(accuracy), "\n")





#-------------------------------------------------------

# Importa dados de teste
validacao <- read.csv("validacao.csv")
treino1 <- read.csv("treino.csv")
treino1 <- treino1[,-c(1,41)]
validacao <- validacao[,-1]
dados<-rbind(validacao, treino1)

trans1 <- preProcess(dados, method = c("pca"))
PC1 <- predict(trans1, dados)

eigenvalues1 <- eigen(cov(PC1))$values
variance_explained1 <- eigenvalues1 / sum(eigenvalues1)
plot(variance_explained1, type = "b", ylab = "Proporção da Variância Explicada", xlab = "Componente Principal", main = "Gráfico dos Autovalores 2")
dados <- PC1[, (1:10)]


#-------------------------------------------------------

#treina os dados 
test <- best_test
train <- best_train
x_train <- treino[train, ]
y_train <- y[train]
x_test <- dados[1:dim(validacao)[1],] #treino[test, ]
y_test <- y[test]


# Calcula as probabilidades para cada classe
p_classes <- numeric(5)
for (class_label in 0:4) {
  p_classes[class_label + 1] <- sum(y_train == class_label) / length(y_train)
}
c1 = x_train[y_train == 0, ]
c2 = x_train[y_train == 1, ]
y_hat <- integer(dim(x_test)[1])
espaco_de_verossimilhanca <- matrix(0, nrow = length(y_test), ncol = 2)  # Inicializa com zeros

for (i in 1:length(x_test)) {
  # Cálculo das densidades para cada classe
  p_values <- numeric(5)
  for (class_label in 0:4) {
    p_values[class_label + 1] <- pdfKDE(x_test[i, ], nrow(x_train[y_train == class_label, ]), x_train[y_train == class_label, ], h_model)
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



#-----------------------------------------------
#-----------------------------------------------
#------------------------ SVM ------------------
#-----------------------------------------------
#-----------------------------------------------
# Calcula as estimativas de densidade para o conjunto de treino completo
pdf_values <- matrix(0, nrow = nrow(x_train), ncol = 5)

for (i in 1:nrow(x_train)) {
  for (class_label in 0:4) {
    pdf_values[i, class_label + 1] <- pdfKDE(x_train[i,], sum(y_train == class_label), x_train[y_train == class_label,], h_model)
  }
}

# Adiciona as estimativas de densidade como features ao conjunto de treino
treino_final <- cbind(x_train, pdf_values)

# Divide novamente em X e Y após adicionar as novas features
x <- treino[, 2:44]
y <- as.matrix(as.numeric(treino[, "y_num"]))

test <- best_test
train <- best_train
x_train <- treino[train, ]
y_train <- y[train]
x_test <- treino[test, ]
y_test <- y[test]


# Define os parâmetros do modelo SVM
svm_model <- svm(treino, y, kernel = "radial", cost = 10)

# Faz as previsões no conjunto de teste
y_svm <- predict(svm_model, treino[test_model,])

# Avaliação do desempenho do SVM
svm_accuracy <- sum(y_test == y_svm) / length(y_test)
print(paste("Acurácia do SVM: ", svm_accuracy))
