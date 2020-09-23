source('E:\\MLDLAlgo\\R\\utils.R')


LR.fit <- function(.self,x,y,show_fig=FALSE,intercept=TRUE){
  
  x.train <- x
  y.train <- y
  
  if(intercept == T){
    
    intercept <<- intercept
    x.train <- cbind(x,1)
    
  }
  W <<- solve(t(x.train) %*% x.train) %*% t(x.train) %*% y.train
  
}

###########################################################
LR.evaluate <- function(.self,x,y){
  
  yhat <- .self$predict(x)
  
  return(MSE(y,yhat))
}

###########################################################
LR.predict <- function(.self,x){
  
 
  if(intercept == T){
    
    x <- cbind(x,1)
  }
  return(x %*% W)
}

###########################################################

LR.init <- function(){
  
  W <- 0
  intercept <- FALSE
  show_fig <- FALSE
  
}
###########################################################
LinearRegression <- setRefClass('LinearRegression', 
                                methods = list(initialize=LR.init,fit=LR.fit,evaluate=LR.evaluate,predict=LR.predict))

