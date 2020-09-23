
## Linear Regression on boston data ##
source('E:/MLDLAlgo/R/utils.R')
source('E:/MLDLAlgo/R/Linear regression/model.R')

boston.data <- get.boston()

X <- boston.data$x
Y <- boston.data$y 

train.idx <- (1:nrow(X))[1:round(0.9*nrow(X))]
test.idx <- (1:nrow(X))[-train.idx]

X.train <- X[train.idx,]
Y.train <- Y[train.idx,]

X.test <- X[test.idx,]
Y.test <- Y[test.idx,]

lr.model <- LinearRegression()

lr.model$fit(X.train,Y.train)

train.acc <- lr.model$evaluate(X.train,Y.train)

cat('MSE score on train data : ',train.acc,'\n')

test.acc <- lr.model$evaluate(X.test,Y.test)

cat('MSE score on test data : ',test.acc)


# MSE score on train data :  23.20597 
# MSE score on test data :  10.77455
