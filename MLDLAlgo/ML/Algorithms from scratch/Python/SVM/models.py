# -*- coding: utf-8 -*-
"""
Created on Fri May  1 08:01:01 2020

@author: Avinash
"""

import sys
import numpy as np

sys.path.append("E:\\MLDLAlgo")
from metrics import Accuracy
import matplotlib.pyplot as plt



class simple_LinearSVM(object):
    
    def margins(self,y_pred):
        
        marg=1-(self.Y*y_pred)
        return marg
        
        
    def loss(self,margins):
        
        margins[margins<0]=0
        return (0.5*(self.Weights.T.dot(self.Weights)))+(self.C*(margins.sum()))
                                                                        
    def fit(self,X,Y,C=0.001,epochs=10,learning_rate=0.001,show_fig=False):
        
        self.X=X
        self.Y=Y
        self.Y[self.Y==0]=-1
        self.Weights=np.random.randn(self.X.shape[1])
        self.bias=0
        self.C=C
        loss=[]
        metric=[]
        for i in range(epochs):
            
            Y_pred=self.X.dot(self.Weights)+self.bias
            marg=self.margins(Y_pred)
            cost=self.loss(marg)
            
            grad_w=self.Weights-self.C*(self.X[marg>0].T.dot(self.Y[marg>0]))
            grad_b=self.C*self.Y[marg>0].sum()
            
            self.Weights-=learning_rate*grad_w
            self.bias-=learning_rate*grad_b
            
            epoch_acc=Accuracy(self.Y,np.sign(Y_pred))
            print("epoch : ",i+1,"Training Accuracy : ",epoch_acc)
            metric.append(epoch_acc*100)
            loss.append(cost)
        if(show_fig):
            
            plt.plot(range(1,epochs+1),metric,label='training accuracy')
            plt.legend()
            plt.show()
            plt.plot(range(1,epochs+1),loss,label='training loss')
            plt.legend()
            plt.show()
            
    def predict(self,x):
        
        Y_pred=x.dot(self.Weights)+self.bias
        Y_pred=np.sign(Y_pred)
        
        return Y_pred
    
    def evaluate(self,x,y):
        
        y[y==0]=-1
        Y_pred=self.predict(x)
        
        return Accuracy(y,Y_pred)
    
    
    
    
    
    
    
    
    
    
    
######################################################################################################################



class MultiClass_LinearSVM(object):
    
    def margins(self,y,y_pred):
        
        marg=1-(y*y_pred)
        #marg[marg<0]=0
        
        return marg
        
        
    def loss(self,w,margins):
        
        margins[margins<0]=0
        return (0.5*(w.T.dot(w)))+(self.C*(margins.sum()))
                                                                        
    def fit(self,X,Y,C=0.001,epochs=10,learning_rate=0.001,show_fig=False):
        
        self.X=X
        self.Y=Y.astype(np.int32)
        self.Weights=np.random.randn(len(set(self.Y)),self.X.shape[1])
        self.bias=np.zeros((len(set(self.Y))))
        self.C=C
        loss=[]
        metric=[]
        for i in range(epochs):
            
            cost=0
            for j in set(self.Y):
                
                grad_w=np.zeros(self.X.shape[1])
                grad_b=0
                for k in [h for h in set(self.Y) if h!=j]:
                    posidx=np.where(self.Y==j)
                    negidx=np.where(self.Y==k)
                    
                    tmp_x=np.concatenate((self.X[posidx],self.X[negidx]),axis=0)
                    tmp_y=np.concatenate((self.Y[posidx],self.Y[negidx]),axis=0)
                    tmp_y[tmp_y==j]=1
                    tmp_y[tmp_y==k]=-1
                    tmp_pred=tmp_x.dot(self.Weights[j])+self.bias[j]
                    
                    marg=self.margins(tmp_y,tmp_pred)
                    cost+=self.loss(self.Weights[j],marg)
                    
                
                    grad_w+=self.Weights[j]-self.C*(tmp_x[marg>0].T.dot(tmp_y[marg>0]))
                    grad_b+=self.C*tmp_y[marg>0].sum()
                    
                self.Weights[j]-=learning_rate*grad_w
                self.bias[j]-=learning_rate*grad_b
                
            Y_pred=self.predict(self.X)
            
            epoch_acc=Accuracy(self.Y,Y_pred)
            print("epoch : ",i+1,"Training Accuracy : ",epoch_acc,'loss : ',cost)
            metric.append(epoch_acc*100)
            loss.append(cost)
            
        if(show_fig):
            
            plt.plot(range(1,epochs+1),metric,label='training accuracy')
            plt.legend()
            plt.show()
            plt.plot(range(1,epochs+1),loss,label='training loss')
            plt.legend()
            plt.show()
            
    def predict(self,x):
        
        bias=self.bias.reshape(1,self.bias.shape[0])
        Y_pred=x.dot(self.Weights.T)+bias
        
        Y_pred=Y_pred.argmax(axis=1)
        
        return Y_pred
    
    def evaluate(self,x,y):
        
        Y_pred=self.predict(x)
        
        return Accuracy(y,Y_pred)
        
    
    
    
######################################################################################################################


class KernelSVM():
    
    
    def rbf(self,x1,x2):
            
        return np.exp(-self.gamma*(np.linalg.norm(x1[np.newaxis,:] - x2[:,np.newaxis])**2))
        
    
    def loss(self):
         
        return np.sum(self.alphas)-(0.5*np.sum(self.YYK*np.outer(self.alphas,self.alphas)))
                
    def fit(self,X,Y,epochs=10,gamma=0.0001,learning_rate=0.0001,C=1.0,kernel='rbf',show_fig=False):
        
        self.X=X
        self.Y=Y
        self.Y[Y==0]=-1
        self.kernel=kernel
        self.C=C
        self.learning_rate=learning_rate
        self.gamma=gamma
        loss=[]
        
        
        self.alphas=np.random.randn(self.X.shape[0])
        self.bias=0
        
        if(self.kernel=='rbf'):
            self.kernel_fun=self.rbf(self.X,self.X)
            
        self.YY=np.outer(self.Y,self.Y)
        self.YYK=self.YY*self.kernel_fun
        
        for i in range(epochs):
            
            self.alphas+=self.learning_rate*(np.ones(self.X.shape[0])-self.YYK.dot(self.alphas))
            self.alphas[self.alphas<0]=0
            self.alphas[self.alphas>self.C]=self.C
            cost=self.loss()
            
            loss.append(cost)
            print("epoch : ",i+1,"loss : ",cost)
        
        validalph_idx=np.where((self.alphas>0) & (self.alphas<self.C))[0]
        
        if(len(validalph_idx)>0):
            if(self.kernel=='rbf'):
                b=self.Y[validalph_idx]-(self.alphas*self.Y).dot(self.rbf(self.X,self.X[validalph_idx]))
            
                self.bias=np.mean(b)
            
        
        if(show_fig==True):
            
            plt.plot(range(1,epochs+1),loss,label='Training loss')
            plt.legend()
            plt.show()
                        
    def predict(self,x):
        
        if(self.kernel=='rbf'):
            Y_pred=((self.alphas*self.Y).dot(self.rbf(self.X,x)))+self.bias
        
        return np.sign(Y_pred)
    
    def Evaluate(self,x,y):
        
        y[y==0]=-1
        Y_pred=self.predict(x)
        
        return Accuracy(self.Y,Y_pred)
        