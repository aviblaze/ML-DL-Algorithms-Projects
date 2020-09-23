# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 20:32:59 2020

@author: Avinash
"""
import sys
sys.path.append("E:\\MLDLAlgo")

import numpy as np
from losses import Binary_Cross_Entropy,Cross_Entropy
from metrics import Accuracy
from utils import sigmoid,softmax
import matplotlib.pyplot as plt

class ML_SimpleLogisticRegression(object):
    
    def fit(self,x,y,epochs=10,learning_rate=0.001,showfig=False):
        
        
        self.X=x
        self.Y=y
        
        self.Weights=np.random.randn(self.X.shape[1])
        self.bias=np.random.randn(1)
        
        loss=[]
        metric=[]
        
        for i in range(epochs):
            
            Y_pred=sigmoid(self.X,self.Weights,self.bias)
            
            self.Weights-=learning_rate*((self.X.T.dot(Y_pred-self.Y))/len(self.X))
            self.bias-=learning_rate*((Y_pred.dot(1-Y_pred).sum())/len(self.X))
            
            
            epoch_loss=Binary_Cross_Entropy(self.Y,Y_pred)
            
            epoch_metric=Accuracy(self.Y,Y_pred)
            
            loss.append(epoch_loss)
            metric.append(epoch_metric)
            
            print("epoch : ",i+1,"loss : ",epoch_loss,"Accuracy : ",epoch_metric)
            
        if(showfig):
                
            plt.plot(range(1,epochs+1),loss,label='Training loss')
            plt.plot(range(1,epochs+1),metric,label='Training metric')
            plt.legend()
            plt.show()
            
    def predict(self,x):
        
        return sigmoid(x,self.Weights,self.bias)            
            
    def evaluate(self,x,y):
        
        Y_pred=sigmoid(x,self.Weights,self.bias)
        
        return Accuracy(y,Y_pred)
    



###########################################


class ML_MultiLogisticRegression(object):
    
    def fit(self,x,y,epochs=10,learning_rate=0.001,showfig=False):
        
        
        self.X=x
        self.Y=y
        
        
        self.Weights=np.random.randn(self.X.shape[1],self.Y.shape[1])
        self.bias=np.random.randn(self.Y.shape[1])
        
        loss=[]
        metric=[]
        
        for i in range(epochs):
            
            Y_pred=softmax(self.X,self.Weights,self.bias)
            

            self.Weights-=learning_rate*((self.X.T.dot(Y_pred-self.Y))/len(self.X))
            self.bias-=learning_rate*((Y_pred.T.dot(1-Y_pred).sum(axis=1))/len(self.X))
        
            
            epoch_loss=Cross_Entropy(self.Y,Y_pred)
            epoch_metric=Accuracy(self.Y,Y_pred)*100
            
            loss.append(epoch_loss)
            metric.append(epoch_metric)
            
            print("epoch : ",i+1,"loss : ",epoch_loss,"Accuracy : ",epoch_metric)
            
        if(showfig):
                
            plt.plot(range(1,epochs+1),loss,label='Training loss')
            plt.plot(range(1,epochs+1),metric,label='Training metric')
            plt.legend()
            plt.show()
            
    def predict(self,x):
        
        return sigmoid(x,self.Weights,self.bias)            
            
    def evaluate(self,x,y):
        
        Y_pred=sigmoid(x,self.Weights,self.bias)
        
        return Accuracy(y,Y_pred)
    



###########################################


