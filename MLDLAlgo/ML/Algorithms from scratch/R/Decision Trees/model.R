library(Dict)
source('E:/MLDLAlgo/R/utils.R')

find.best.feature <- function(x,y){
  
  best.ig <- -1e6
  best.feature <- NULL
  val <- NULL
  
  
  for(i in 1:dim(x)[2]){
    
    if(class(x[,i]) == 'character' | length(table(x[,i])) <= 10){
      ig <- Information.Gain(x[,i],y,'f')
    }else{
      ig <- Information.Gain(x[,i],y,'n')
    }
    
    if(as.double(ig['ig']) > best.ig){
      
      best.ig <- as.double(ig['ig'])
      best.feature <- i
      val <- as.double(ig['best.var'])
    }
  }
  
  res <- c(best.feature,val)
  names(res) <- c('feat','val')
  return(res)
}

node <- function(.self,x,y,side=NULL,parent=NULL,d=NULL){
  
  
  if(d <= Gdepth){
    
    #cat('currentDepth : ',d,'\n')
    best.feature.details <- find.best.feature(x,y)
    
    feat <- best.feature.details['feat']
    val <- best.feature.details['val']
    
    if(class(val) == 'character'){
      
      feat <- as.numeric(feat)
      if(is.null(side) & is.null(parent)){
        
        Tree['root'] <- paste(c(feat,val),collapse = ',')
        Tree[paste(c(feat,val),collapse = ',')]=list('left'=NULL,'right'=NULL)
      }else{
        
        Tree[parent][side] <- paste(c(feat,val),collapse = ',')
        Tree[paste(c(feat,val),collapse = ',')]=list('left'=NULL,'right'=NULL)
      }
      
      left.idx <- which(x[,feat] == val)
      right.idx <- (1:nrow(x))[-left.idx]
      
      
      left.tmp.x <- x[left.idx,]
      left.tmp.y <- y[left.idx]
      right.tmp.x <- x[right.idx,]
      right.tmp.y <- y[right.idx]
      
      if(is.null(nrow(left.tmp.x))){
        left.tmp.x <- matrix(left.tmp.x,nrow=1)
      }
      if(is.null(nrow(right.tmp.x))){
        right.tmp.x <- matrix(right.tmp.x,nrow=1)
      }
      
      if(Entropy(y[left.idx]) == 0 | length(left.idx) < GminSplit ){
        
        Tree[paste(c(feat,val),collapse = ',')]['left']=as.numeric(names(sort(table(y[left.idx]),decreasing = T))[1])
        if(length(right.idx) == nrow(x)){
          Tree[paste(c(feat,val),collapse = ',')]['right']=as.numeric(names(sort(table(y[right.idx]),decreasing = T))[1])  
        }else{
          .self$node(right.tmp.x,right.tmp.y,'right',paste(c(feat,val),collapse = ','),d+1)
        }
        
      }else if(Entropy(y[right.idx]) == 0 | length(right.idx) < GminSplit  ){
        
        Tree[paste(c(feat,val),collapse = ',')]['right']=as.numeric(names(sort(table(y[right.idx]),decreasing = T))[1])
        if(length(left.idx) == nrow(x)){
          Tree[paste(c(feat,val),collapse = ',')]['left']=as.numeric(names(sort(table(y[left.idx]),decreasing = T))[1])  
        }else{
          .self$node(left.tmp.x,left.tmp.y,'left',paste(c(feat,val),collapse = ','),d+1)
        }
      }else{
        
        .self$node(left.tmp.x,left.tmp.y,'left',paste(c(feat,val),collapse = ','),d+1)
        .self$node(right.tmp.x,right.tmp.y,'right',paste(c(feat,val),collapse = ','),d+1)
      }
    }else{
      
      
      if(is.null(side) & is.null(parent)){
        
        Tree['root'] <- paste(c(feat,val),collapse = ',')
        Tree[paste(c(feat,val),collapse = ',')]=list('left'=NULL,'right'=NULL)
      }else{
        
        Tree[parent][side] <- paste(c(feat,val),collapse = ',')
        Tree[paste(c(feat,val),collapse = ',')]=list('left'=NULL,'right'=NULL)
      }
      
      left.idx <- which(x[,feat] <= val)
      right.idx <- (1:nrow(x))[-left.idx]
     
      left.tmp.x <- x[left.idx,]
      left.tmp.y <- y[left.idx]
      right.tmp.x <- x[right.idx,]
      right.tmp.y <- y[right.idx]

      if(Entropy(y[left.idx]) == 0 | length(left.idx) < GminSplit ){
        if(length(y[left.idx]) ==0 ){
          Tree[paste(c(feat,val),collapse = ',')]['left']=as.numeric(names(sort(table(y[right.idx])))[1])
        }else{
          Tree[paste(c(feat,val),collapse = ',')]['left']=as.numeric(names(sort(table(y[left.idx]),decreasing = T))[1])
        }
        if(length(right.idx) == nrow(x) | length(right.idx) < GminSplit){
          Tree[paste(c(feat,val),collapse = ',')]['right']=as.numeric(names(sort(table(y[right.idx]),decreasing = T))[1])  
        }else{
          .self$node(right.tmp.x,right.tmp.y,'right',paste(c(feat,val),collapse = ','),d+1)
        }
        
      }else if(Entropy(y[right.idx]) == 0 | length(right.idx) < GminSplit  ){
        
        if(length(right.idx)==0){
          Tree[paste(c(feat,val),collapse = ',')]['right']=as.numeric(names(sort(table(y[left.idx])))[1])
        }else{
          Tree[paste(c(feat,val),collapse = ',')]['right']=as.numeric(names(sort(table(y[right.idx]),decreasing = T))[1]) 
        }
        if(length(left.idx) < GminSplit ){
          Tree[paste(c(feat,val),collapse = ',')]['left']=as.numeric(names(sort(table(y[left.idx]),decreasing = T))[1])  
        }else{
        .self$node(left.tmp.x,left.tmp.y,'left',paste(c(feat,val),collapse = ','),d+1)
        }
      }else{
        
        .self$node(right.tmp.x,right.tmp.y,'right',paste(c(feat,val),collapse = ','),d+1)
        .self$node(left.tmp.x,left.tmp.y,'left',paste(c(feat,val),collapse = ','),d+1)
        
      }
    }
  }else{
    Tree[parent][side] <- as.numeric(names(sort(table(y)))[1])
  }
}

construct.Tree <- function(.self,x,y){
  
  .self$node(x,y,d=0)
  print('Decision Tree for training data has been generated.')
}

fit <- function(.self,x,y,depth,min.split){
  
  Gdepth <<- 3
  GminSplit <<- 9
  Tree <<- dict('root'=NA)
  
  
  Gdepth <<- depth
  GminSplit <<- min.split
  
  .self$construct.Tree(x,y)
  
}

predict <- function(x){
  
  yhat <- vector()
  for(i in 1:nrow(x)){
    
    root.node.details <- Tree['root']
    
    tmp <- as.double(unlist(strsplit(unlist(root.node.details),split = ',')))
   
    feat <- tmp[1]
    val <- tmp[2]
    
    if(class(x[,feat]) == 'character'){
      
      if(x[i,feat] == root.node.details[2]){
        choose.node <- Tree[root.node.details]['left']
        choose.node <- unlist(strsplit(choose.node,split = ','))
        feat <- as.double(choose.node[1])
        if(!is.na(as.double(choose.node[2]))){
          val <- as.double(choose.node[2])
        }
      }else{
        choose.node <- Tree[root.node.details]['right']
        choose.node <- unlist(strsplit(choose.node,split = ','))
        feat <- as.double(choose.node[1])
        if(!is.na(as.double(choose.node[2]))){
          val <- as.double(choose.node[2])
        }
      }
    }else{
    
      if(x[i,feat] <= val){
        choose.node <- Tree[root.node.details]['left']
        
      }else{
    
        choose.node <- Tree[root.node.details]['right']
      }
    }
  
    while(grepl(',',choose.node)){
     
      tmp <- as.double(unlist(strsplit(unlist(choose.node)[1],split = ',')))
      
      feat <- tmp[1]
      val <- tmp[2]
      if(class(x[i,feat]) == 'character'){
        print('in if')
        if(x[i,feat] == val){
          choose.node <- Tree[choose.node]['left']
          tmp <- strsplit(choose.node,split = ',')
          feat <- as.double(tmp[1])
          if(!is.na(as.double(tmp[2]))){
            val <- as.double(tmp[2])
          }
        
        }else{
          choose.node <- Tree[choose.node]['right']
          choose.node <- strsplit(choose.node,split = ',')
          feat <- as.double(choose.node[1])
          if(!is.na(as.double(choose.node[2]))){
            val <- as.double(choose.node[2])
          }
          
        }
      }else{
      
        if(x[i,feat] <= val){
          
          choose.node <- Tree[unlist(choose.node)]['left']
        }else{
          
          choose.node <- Tree[unlist(choose.node)]['right']
        }
      }
    }
    
    yhat <- c(yhat,as.numeric(unlist(choose.node)))
  }

  return(yhat)
}

evaluate <- function(.self,x,y){
  
  
  ypred <- .self$predict(x)
  ypred[1:5]
  return(Accuracy(y,ypred))
}

DecisionTree <- setRefClass('DecisionTree',
                             methods=list(find.best.feature=find.best.feature,node=node,
                                          construct.Tree=construct.Tree,fit=fit,predict=predict,evaluate=evaluate))