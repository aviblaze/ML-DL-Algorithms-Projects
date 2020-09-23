source('E:/MLDLAlgo/R/Logistic Regression/model.R')
source('E:/MLDLAlgo/R/utils.R')

mnist.data <- get_binary_mnist()

X <- mnist.data$x
Y <- mnist.data$y

train.idx <- sample(1:nrow(X),round(0.9*nrow(X)))
test.idx <- (1:nrow(X))[-train.idx]

X.train <- X[train.idx,]
Y.train <- as.matrix(Y[train.idx,])

X.test <- X[test.idx,]
Y.test <- as.matrix(Y[test.idx,])


model <- LogisticRegression()

model$fit(X.train,Y.train,epochs=10,learning_rate=0.0001)

model$evaluate(X.test,Y.test)