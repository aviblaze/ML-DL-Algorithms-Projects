# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:10:05 2020

@author: Avinash
"""

import numpy as np

def RMSE(Y,Yhat):
    
    return (np.sqrt(((Y-Yhat)**2).sum()/len(Y)))


############################################################################################################

def Accuracy(Y,Yhat):
    
    
    
    if(Y.ndim==1):
        if(Yhat.max()<1):
            Yhat[Yhat>=0.5]=1
            Yhat[Yhat<0.5]=0
            return len(Yhat[Y==Yhat])/len(Y)
        else:
            #print("in else")
            return len(Yhat[Y==Yhat])/len(Y)

    Y_pred=np.zeros((len(Y),Y.shape[1]))
    Y_pred[range(len(Y_pred)),np.argmax(Yhat,axis=1)]=1

    return len(Y_pred[(np.sum(np.abs(Y_pred-Y),axis=1))==0])/len(Y)


############################################################################################################

def Entropy(y):
    
    ent=0
    
    for _class in set(y):  
        samp=len(y[y==_class])/len(y)
        ent-=samp*np.log2(samp)        
    
    return ent
        
        



############################################################################################################