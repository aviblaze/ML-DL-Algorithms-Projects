

MSE <- function(Y,Yhat){
  
  return(mean((Y-Yhat)^2))
}

########################################################################

get_binary_mnist <- function(){
  
  mnist <- read.csv('E:/AI and ML/Mnist Dataset/train.csv')
  mnist <- mnist[mnist$label==0 | mnist$label==1,]
  
  rownames(mnist) <- 1:nrow(mnist)
 
  ##shuffle data
  mnist <- mnist[sample(as.numeric(rownames(mnist))),] 
  
  
  X <- mnist[,!colnames(mnist) %in% c('label')] 
  Y <- mnist$label

  return(list(x=as.matrix(X),y=as.matrix(Y)))
}

########################################################################

get.boston <- function(){
  
  boston.data <- read.csv('E:/MLDLAlgo/boston.csv')
  dat.rows <- nrow(boston.data) 
    
  boston.data <- as.numeric(unlist(apply(boston.data,1,FUN=strsplit,split=' +',fixed=F)))
  boston.data <- boston.data[!is.na(boston.data)]
  
  boston.data <- matrix(boston.data,byrow=T,dat.rows)
  
  X <- boston.data[,1:13]
  Y <- matrix(boston.data[,14])
  
  return(list(x=X,y=Y))
}

########################################################################

sigmoid <- function(w,x,b){
  
  return(1/(1+exp(-((x %*% w)+b+0.0000000001))))
}

########################################################################

Accuracy <- function(y,yhat){
  
  #cat(dim(y),dim(yhat),class(y),class(yhat))
  return(mean(yhat==y))
}

########################################################################

CrossEntropy <- function(y,yhat){
  
  return(-mean(y*log(yhat+0.000000001)+((1-y)*log(1-yhat+0.000000001))))
  
}

########################################################################

Entropy <- function(y){
  
  p <- table(y)/length(y)
  
  return(-sum(p * log2(p)))
}

########################################################################

Information.Gain <- function(x,y,type){
  
  tot.ent <- Entropy(y)
  
  if(tolower(type) == 'f' & length(table(x))<=10){
    
    fac <- table(x)
    fac.names <- names(fac)
    
    if(class(x) != 'character'){
      fac.names <- as.integer(names(fac))
    }
    
    best.ent <- NULL
    
    best.ent.val <- 1e10
    weigh.ent <- 0
    for(i in fac.names){
      
      ent <- Entropy(y[x==i])
      
      weigh.ent <- weigh.ent + ( (length(y[x==i])/length(y)) * ent )
      
      if(ent < best.ent.val){
        best.ent <- i
        best.ent.val <- ent
      }
    }
    
    res <- c(tot.ent - weigh.ent,best.ent)
    names(res) <- c('ig','best.var') 
    return(res)
    
  }else if(tolower(type)=='n'){
    
    val <- mean(sort(x))
    weigh.ent <- 0
    
    
    ent1 <- Entropy(y[x <= val])
    weigh.ent <- weigh.ent + ((length(y[x <= val])/length(y)) * ent1)
    
    ent2 <- Entropy(y[x > val])
    weigh.ent <- weigh.ent + ((length(y[x > val])/length(y)) *ent2)
    
    res <- c(tot.ent - weigh.ent,val)
    names(res) <- c('ig','best.var') 
    return(res)
  }else{
    print('supports only data of type "chatacter"(upto 10 factor levels) and "numeric".')
  }
  
}
