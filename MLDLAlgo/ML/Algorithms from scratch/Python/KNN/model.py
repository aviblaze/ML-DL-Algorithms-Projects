# -*- coding: utf-8 -*-
"""
Created on Sat May  2 07:28:19 2020

@author: Avinash
"""

import sys
import numpy as np

sys.path.append("E:\\MLDLAlgo")

from metrics import Accuracy

class KNN(object):
    
    def fit(self,X,Y,n=3):
        
        self.X=X
        self.Y=Y.astype(np.int32)
        self.N=n
    
    def predict(self,X):
        
        Y_pred=np.zeros(X.shape[0])
        
        for i in range(len(X)):
            
            dist_mat=np.sqrt(np.sum((X[i]-self.X)**2,axis=1))
            #print(dist_mat.shape)
            near_neigh=np.argsort(dist_mat)[:self.N]
            #print(near_neigh)
            Y_pred[i]=np.bincount(self.Y[near_neigh]).argmax()
            
        return Y_pred
    
    def evaluate(self,X,Y):
        
        Y_pred=self.predict(X)
        
        return Accuracy(Y,Y_pred)