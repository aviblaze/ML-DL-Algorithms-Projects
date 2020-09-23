source('E:/MLDLAlgo/R/utils.R')

boston.data <- get.boston()

X <- boston.data$x
Y <- boston.data$y 

train.idx <- (1:nrow(X))[1:round(0.9*nrow(X))]
test.idx <- (1:nrow(X))[-train.idx]

X.train <- X[train.idx,]
Y.train <- Y[train.idx,]

X.test <- X[test.idx,]
Y.test <- Y[test.idx,]

train <- data.frame(cbind(X.train,Y.train))
colnames(train) <- c(1:13,'price')
test <- data.frame(cbind(X.test,Y.test))
colnames(test) <- c(1:13,'price')

model <- lm(price ~ .,data=train)

#print(summary(model))

pred <- predict(model,test)

cat('MSE of test data : ',MSE(Y.test,pred))



# MSE of test data :  10.77455