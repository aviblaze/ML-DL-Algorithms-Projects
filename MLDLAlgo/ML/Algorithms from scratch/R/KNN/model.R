source('E:/MLDLAlgo/R/utils.R')

initialize <- function(){
  
  X <- 0
  Y <- 0
  K <- 1
}

fit <- function(x,y,k){
    X <<- x
    Y <<- y
    K <<- k
}

predict <- function(x){
  
  yhat <- vector()
  
  for(i in x){
    norm <- sqrt(apply(sweep(X,2,i)^2,1,sum))
    
    names(norm) <- 1:nrow(X)
    
    
    neigh <- as.integer(names(sort(norm,decreasing = T))[1:K])
    
  yhat <- c(yhat,as.integer(names(sort(table(Y[neigh]),decreasing = T)))[1]) 
  
  }
  
  return(yhat)
  
}


evaluate <- function(.self,x,y){
  
  ypred <- .self$predict(x)
  
  cat('Test Accuracy : ', Accuracy(y,yhat))
}

KNN <- setRefClass('KNN',
            methods = list(fit=fit,predict=predict,evaluate = evaluate))