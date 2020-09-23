# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:26:14 2020

@author: Avinash
"""
import numpy as np

def MSE(Y,Yhat):
    
    return (((Y-Yhat)**2).sum()/len(Y))

def MAE(Y,Yhat):
    
    return (np.abs(Y-Yhat).sum())/len(Y)

def Binary_Cross_Entropy(Y,Yhat):
    
    
    return -(np.log(((1-Yhat[Y==0]).sum()+np.log(Yhat[Y==1]).sum())+1e-7))/len(Y)
    

def Cross_Entropy(Y,Yhat):
    
    return -(np.log(((Y*Yhat)+1e-7)).sum()/len(Y))