#exercicio KDE

rm(list=ls())

# define a função KDE
pdfKDE <- function(xi, N, x)
{
  sum <- 0
  h <- 1.06 * sd(x) * N^(-1/5)
  for (i in 1:N)
  {
    sum <- sum + exp(-(1/(2*h^2) * (t(x[i, ] - xi) %*% (x[i, ] - xi))))
  }
  p <- (1/(N * (sqrt(2 * pi * h))^N)) * sum
  return (p)
}

#validação cruzada com 10 folds usando bayes
bayes_classifier <- function(x_train, y_train, x_test){
  pC1 = (nrow(x_train[y_train == 1, ])) / nrow(x_train)
  pC2 = 1-pC1
  c1 = x_train[y_train == 0, ]
  c2 = x_train[y_train == 1, ]
  y_hat <- c()
  for(i in 1:nrow(x_test)){
    p1 <- pdfKDE(x_test[i, ], nrow(c1),
                 c1)
    p2 <- pdfKDE(x_test[i, ], nrow(c2),
                 c2)
    if(p1*pC1/(p2*pC2) >= 1){
      y_hat <-c(y_hat, 0)
    }
    else{
      y_hat <- c(y_hat, 1)
    }
  }
  return(y_hat)
}

#utiliza-se  a bibl mlbench
library(mlbench)
data <- mlbench.spirals(200, sd = 0.05)
x <- cbind(as.matrix(data[["x"]]))
y <- (as.matrix(as.numeric(data[["classes"]]))-1)
index <- sample(1:nrow(x), length(1:nrow(x)))

plot(x[y==1,1], x[y == 1, 2], col = 'red', xlim = c(-1.5, 1.5), ylim = c(-1.5,1.5), xlab = '', ylab= '',main = "Dados de entrada.")
par(new=T)
plot(x[y==0,1], x[y == 0, 2], col = 'blue', xlim = c(-1.5, 1.5), ylim = c(-1.5,1.5), xlab = '', ylab= '')

# obtem-se acurácia obtido para cada iteração, o desvio padrão das acurácias e a média das acurácias
accuracy <- matrix(nrow = 10, ncol = 1)
j <- 1
best <- 0
for(i in seq(20, 200, 20)){
  test <- index[(i-19):i]
  train <- index[-index[(i-19):i]]
  x_train <- x[train, ]
  y_train <- y[train, ]
  x_test <- x[test, ]
  y_test <- y[test, ]
  y_hat <- bayes_classifier(x_train, y_train, x_test)
  aux <- sum((y_test == y_hat)*1)/20
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

#treina os dados 
test <- best_test
train <- best_train
x_train <- x[train, ]
y_train <- y[train, ]
x_test <- x[test, ]
y_test <- y[test, ]

y_hat <- matrix(nrow = length(y_test), ncol = 1)
espaco_de_verossimilhanca <- matrix(nrow = length(y_test), ncol = 2)

pC1 = (nrow(x_train[y_train == 1, ])) / nrow(x_train)
pC2 = 1-pC1
c1 = x_train[y_train == 0, ]
c2 = x_train[y_train == 1, ]
y_hat <- c()
for(i in 1:nrow(x_test)){
  p1 <- pdfKDE(x_test[i, ], nrow(c1),
               c1)
  p2 <- pdfKDE(x_test[i, ], nrow(c2),
               c2)
  
  espaco_de_verossimilhanca[i, 1] <- p1
  espaco_de_verossimilhanca[i, 2] <- p2
  
  K = (p1 * pC1) / (p2 * pC2)
  
  y_hat[i] <- if (K >= 1) 0 else 1
}

# Plota teste no espaço de verossimilhança
plot(espaco_de_verossimilhanca[y_hat == 0, 1], espaco_de_verossimilhanca[y_hat == 0, 2], col = 'blue', xlim = c(0,1e-06), ylim = c(0,1e-06), xlab = '', ylab = '',main = "Verossimilhanças para o fold 1")
par(new=T)
plot(espaco_de_verossimilhanca[y_hat == 1, 1], espaco_de_verossimilhanca[y_hat == 1, 2], col = 'red', xlim = c(0,1e-06), ylim = c(0,1e-06), xlab = '', ylab = '',main = "Verossimilhanças para o fold 1")


# Dividir amostras de treino para C1 e C2
x_train_C1 <- x_train[y_train == 0, ]
y_train_C1 <- y_train[y_train == 0]
x_train_C2 <- x_train[y_train == 1, ]
y_train_C2 <- y_train[y_train == 1]

# Superficie de densidade de probabilidade
x1seq <- seq(-2.5,2.5, 0.1)
x2seq <- seq(-2.5,2.5, 0.1)
M1 <- matrix(nrow = length(x1seq), ncol = length(x2seq))
M2 <- matrix(nrow = length(x1seq), ncol = length(x2seq))
for(i in 1:length(x1seq))
{
  for(j in 1:length(x2seq))
  {
    x1 <-x1seq[i]
    x2 <- x2seq[j]
    x_in <-as.vector(cbind(x1, x2))
    pdf_c1 <- pdfKDE(x_in, length(y_train_C1), x_train_C1)
    pdf_c2 <- pdfKDE(x_in, length(y_train_C2), x_train_C2)
    
    M1[i, j] = pdf_c1
    M2[i, j] = pdf_c2
  }  
}

plot(x_train[y_train==1,1], x_train[y_train == 1, 2], col = 'red', xlim = c(-1.5, 1.5), ylim = c(-1.5,1.5), xlab = '', ylab= '',main = "Superfíıcie de densidade de probabilidade para o fold 1.")
par(new=T)
plot(x_train[y_train==0,1], x_train[y_train == 0, 2], col = 'blue', xlim = c(-1.5, 1.5), ylim = c(-1.5,1.5), xlab = '', ylab= '')
par(new=T)
contour(x1seq, x2seq, M1, xlim = c(-1.5,1.5), ylim = c(-1.5,1.5), xlab = '', ylab='')
par(new=T)
contour(x1seq, x2seq, M2, xlim = c(-1.5,1.5), ylim = c(-1.5,1.5), xlab = '', ylab='')


# Plota treinamento e teste antes do fod 1
plot(x_train[y_train==1,1], x_train[y_train == 1, 2], col = 'red', xlim = c(-1.5, 1.5), ylim = c(-1.5,1.5), xlab = '', ylab= '',main = "Amostras antes da classificação para o fold 1.")
par(new=T)
plot(x_train[y_train==0,1], x_train[y_train == 0, 2], col = 'blue', xlim = c(-1.5, 1.5), ylim = c(-1.5,1.5), xlab = '', ylab= '')
par(new=T)
plot(x_test[,1], x_test[, 2], col = 'black', xlim = c(-1.5, 1.5), ylim = c(-1.5,1.5), xlab = '', ylab= '')


# Plota treinamento e teste apos fold 1
plot(x_train[y_train==1,1], x_train[y_train == 1, 2], col = 'red', xlim = c(-1.5, 1.5), ylim = c(-1.5,1.5), xlab = '', ylab= '',main = "Amostras após a classificação para o fold 1.")
par(new=T)
plot(x_train[y_train==0,1], x_train[y_train == 0, 2], col = 'blue', xlim = c(-1.5, 1.5), ylim = c(-1.5,1.5), xlab = '', ylab= '')
par(new=T)
plot(x_test[y_test==0,1], x_test[y_test==0, 2], col = 'green', xlim = c(-1.5, 1.5), ylim = c(-1.5,1.5), xlab = '', ylab= '')
par(new=T)
plot(x_test[y_test==1,1], x_test[y_test==1, 2], col = 'yellow', xlim = c(-1.5, 1.5), ylim = c(-1.5,1.5), xlab = '', ylab= '')

plot(x_train[y_train==1,1], x_train[y_train == 1, 2], col = 'red', xlim = c(-1.5, 1.5), ylim = c(-1.5,1.5), xlab = '', ylab= '',main = "Superfíıcie de densidade de probabilidade para o fold 1.")
par(new=T)
plot(x_train[y_train==0,1], x_train[y_train == 0, 2], col = 'blue', xlim = c(-1.5, 1.5), ylim = c(-1.5,1.5), xlab = '', ylab= '')
par(new=T)
contour(x1seq, y_hat, xlim = c(-1.5,1.5), ylim = c(-1.5,1.5), xlab = '', ylab='')
