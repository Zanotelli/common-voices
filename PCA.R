# Limpa o ambiente de trabalho
rm(list=ls())

# Carrega pacotes necessários
library(RnavGraphImageData)
library(caret)
library(e1071)
library(RSNNS)
library(yardstick)

# Define a semente para reprodução
set.seed(1)

# Carrega a base de dados de faces Olivetti
data(faces)
faces <- t(faces)

# Define uma função para mostrar uma imagem da base de dados
MostraImagem <- function(x) {
  rotate <- function(x) t(apply(x, 2, rev))
  img <- matrix(x, nrow=64)
  cor <- rev(gray(50:1/50))
  image(rotate(img), col=cor)
}

# Mostra a primeira imagem da base de dados
MostraImagem(faces[1, ])

# Renomeia as colunas da matriz 'faces'
nomeColunas <- NULL
for(i in 1:ncol(faces)) {
  nomeColunas <- c(nomeColunas, paste("a", as.character(i), sep="."))
}
colnames(faces) <- nomeColunas
rownames(faces) <- NULL

# Aplica pré-processamento (BoxCox, center, scale, pca) nos dados
trans <- preProcess(faces, method = c("BoxCox", "center", "scale", "pca"))
PC <- predict(trans, faces)

# Seleciona as 10 primeiras componentes principais
faces <- PC[, (1:10)]

# Cria rótulos para os dados
y <- NULL
for(i in 1:nrow(faces)) {
  y <- c(y, ((i-1) %/% 10) + 1)
}

# Separa os dados em características (x) e rótulos (y)
x <- faces

# Loop para diferentes proporções de teste/treinamento
for (q in 1:9) {
  # Inicializa matriz para armazenar acurácias
  accuracy_vec <- matrix(nrow = 10, ncol = 1)
  
  # Loop para validação cruzada
  for (k in 1:10) {
    # Embaralha os dados
    index <- sample(1:nrow(x), length(1:nrow(x)))
    x <- x[index, 1:ncol(x)]
    y <- y[index]
    
    # Divide os dados em treinamento e teste
    xy_all <- splitForTrainingAndTest(x, y, ratio = (q/10))
    x_train <- xy_all$inputsTrain
    y_train <- xy_all$targetsTrain
    x_test <- xy_all$inputsTest
    y_test <- xy_all$targetsTest
    
    # Treina um classificador de Bayes ingênuo
    model <- naiveBayes(x_train, y_train)
    
    # Faz previsões no conjunto de teste
    y_hat <- predict(model, x_test)
    
    # Calcula a acurácia
    accuracy <- mean(y_hat == y_test)
    accuracy_vec[k,] <- accuracy
  }
  
  # Exibe estatísticas de acurácia
  print("Acurácia:")
  print(q)
  print(accuracy_vec * 100)
  print(mean(accuracy_vec))
  print(sd(accuracy_vec))
  
  # Cria e exibe a matriz de confusão
  confusao = table(y_test, y_hat)
  print("Matriz de Confusão:")
  print(confusao)
  image(confusao)
  
  # Bibliotecas adicionais
  library(ggplot2)
  library(yardstick)
} 
# Comentado para evitar execução desnecessária
# df = data.frame(y_hat, y_test)
# cmat = conf_mat(df, y_test, y_hat)
# autoplot(cmat, type = "heatmap") + scale_fill_gradient(low = "pink", high = "cyan")

# Calcula os autovalores
eigenvalues <- eigen(cov(PC))$values
# Calcula a proporção da variância explicada
variance_explained <- eigenvalues / sum(eigenvalues)
# Gera o gráfico dos autovalores
plot(variance_explained, type = "b", ylab = "Proporção da Variância Explicada", xlab = "Componente Principal", main = "Gráfico dos Autovalores")
