# Limpa o ambiente de trabalho
rm(list=ls())
#Bibliotecas
library(caret)
#Importa CSVs
treino <- read.csv("treino.csv")

#Transforma colunas da TARGET de texto para numeros
treino$y <- factor(treino$y, levels = c('teens', 'twenties', 'thirties', 'fourties', 'fifties'))
treino$y_num <- as.numeric(treino$y)
treino <- treino[, -which(names(treino) == 'y')]

#--------------------Implementa Pre-Processamento - NORMALIZAÇÃO e REMOÇÃO DE RUIDO
# (nao fez diferença)
# Crie o objeto preProcess usando o método "center" e "scale" (normalização) no conjunto de treinamento
preproc <- preProcess(treino, method = c("center", "scale", "knnImpute"))
# Aplique a transformação no conjunto de treinamento
treino_normalizado <- predict(preproc, treino)
#-----------------------------------------------

#--------------------Implementa Pre-Processamento - PCA
#Diminui a dimensionalidade e Verifica quais features sao mais discriminativas
trans <- preProcess(treino_normalizado, method = c("pca"))
PC <- predict(trans, treino_normalizado)
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