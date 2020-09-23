source('E:/MLDLAlgo/R/utils.R')
source('E:/MLDLAlgo/R/Decision Trees/model.R')

mnist.data <- get_binary_mnist()

X <- mnist.data$x
Y <- mnist.data$y

train.idx <- sample(1:nrow(X),round(0.9*nrow(X)))
test.idx <- (1:nrow(X))[-train.idx]

X.train <- X[train.idx,]
Y.train <- as.matrix(Y[train.idx,])

X.test <- X[test.idx,]
Y.test <- as.matrix(Y[test.idx,])


model <- DecisionTree()

model$fit(X.train,Y.train,5,10)

train.acc <- model$evaluate(X.train,Y.train)
cat('Train Accuracy : ',train.acc,'\n')

test.acc <- model$evaluate(X.test,Y.test)
cat('Test Accuracy : ',test.acc)


# [1] "Decision Tree for training data has been generated."
# Train Accuracy :  0.9906731 
# Test Accuracy :  0.9909297
