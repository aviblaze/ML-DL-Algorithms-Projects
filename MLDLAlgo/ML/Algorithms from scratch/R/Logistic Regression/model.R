source('E:/MLDLAlgo/R/utils.R')

initialize <-function(){
  
  W <- 0
  b <- 0
}

fit <- function(.self,X,Y,epochs=10,learning_rate=0.1){
  
  X <- X
  Y <- Y
  lr <- learning_rate
  W <<- rnorm(dim(X)[2])
  b <<- 0
  
  for(i in 1:epochs){
    
    y.pred <- .self$predict(X)
    
    W <<- W - lr*(t(X) %*% (y.pred - Y))
    b <<- b - lr*sum(y.pred - Y)
    
    loss <- CrossEntropy(Y,y.pred)
    
    yhat <- y.pred
    
    yhat[yhat >= 0.5] <- 1
    yhat[yhat < 0.5] <- 0
    
    cat('Epoch : ',i,' Loss : ',loss,' Accuracy : ',Accuracy(Y,yhat))
  }
}

predict <- function(x){
  
  return(sigmoid(W,x,b))
}

evaluate <- function(.self,x,y){
  
  yhat <- .self$predict(x)
  loss <- CrossEntropy(y,yhat) 
  
  yhat[yhat > 0.5] <- 1
  yhat[yhat < 0.5] <- 0
  
  cat('Test Loss : ',loss,'Test Accuracy : ',Accuracy(y,yhat),"\n")
  
}


LogisticRegression <- setRefClass('LogisticRegression',
                                  methods = list(initialize=initialize,fit=fit,predict=predict,evaluate=evaluate))