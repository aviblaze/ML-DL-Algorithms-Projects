# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:17:30 2020

@author: Avinash
"""
import sys
sys.path.append("E:\\MLDLAlgo")


import numpy as np 
from metrics import Accuracy
from scipy.stats import multivariate_normal as MVN


class Gaussian_NaiveBayes(object):
    
    def fit(self,x,y):
        
        self.X=x
        self.Y=y
        self.Class_prob={}
        classes=[int(i) for i in set(self.Y)]
        
        
        if(self.X.ndim==1):
            self.X=self.X.reshape(len(self.X),1)
            
        self.Mean=np.zeros((int(max(classes)+1),self.X.shape[1]))
        self.Var=np.zeros((int(max(classes)+1),self.X.shape[1]))
        
        for i in classes:
            
            tmp_x=self.X[self.Y==i]
            
            self.Mean[i,:]=tmp_x.mean(axis=0)
            self.Var[i,:]=tmp_x.var(axis=0)+1e-2
            self.Class_prob[i]=len(self.Y[self.Y==i])/len(self.Y)
            
        
        
    def predict(self,x):
        
        Y_pred=np.zeros((len(x),int(max(set(self.Y)))+1))
        for i in [int(i) for i in set(self.Y)]:
            
            Y_pred[:,i]=MVN.logpdf(x,mean=self.Mean[i],cov=self.Var[i])+np.log(self.Class_prob[i])
            
        Y_pred=(Y_pred.argmax(axis=1))   
        
        return Y_pred
    
    def evaluate(self,x,y):
        
        Y_pred=self.predict(x)
        
        return Accuracy(y,Y_pred)
            
            
        
        
        
        