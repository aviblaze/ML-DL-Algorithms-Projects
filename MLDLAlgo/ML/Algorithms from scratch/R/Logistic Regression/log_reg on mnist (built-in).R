source('E:/MLDLAlgo/R/utils.R')

mnist.data <- get_binary_mnist()

X <- mnist.data$x
Y <- mnist.data$y

colnames(Y) <- c('label')

mnist.data <- data.frame(cbind(X,Y))

train.idx <- sample(1:nrow(mnist.data),round(0.9*nrow(mnist.data)))
test.idx <- (1:nrow(mnist.data))[-train.idx]

mnist.train <- mnist.data[train.idx,]
mnist.test <- mnist.data[test.idx,]

model <- glm(label ~ ., family = binomial( link = 'logit') ,data=mnist.train)
mnist.predict <- predict(model,mnist.test,type = 'response')

mnist.pred <- ifelse(mnist.predict>0.5,1,0)

cat("Test Accracy : ",Accuracy(mnist.test$label,mnist.pred))