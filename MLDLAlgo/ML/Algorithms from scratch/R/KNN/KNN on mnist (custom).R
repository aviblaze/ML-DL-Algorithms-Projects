source('E:/MLDLAlgo/R/KNN/model.R')
source("E:/MLDLAlgo/R/utils.R")

mnist.data <- get_binary_mnist()

X <- mnist.data$x
Y <- mnist.data$y 

train.idx <- (1:nrow(X))[1:round(0.9*nrow(X))]
test.idx <- (1:nrow(X))[-train.idx]

X.train <- X[train.idx,]
Y.train <- Y[train.idx,]

X.test <- X[test.idx,]
Y.test <- Y[test.idx,]

model <- KNN()

model$fit(X.train,Y.train,3)

train.acc <- model$evaluate(X.train,Y.train)

cat('Train Accuracy : ',train.acc,'\n')

test.acc <- model$evaluate(X.test,Y.test)

cat('Test Accuracy : ',test.acc,'\n')