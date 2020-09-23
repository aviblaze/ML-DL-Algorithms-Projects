# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:10:26 2020

@author: Avinash
"""
import sys
sys.path.append("E:MLDLAlgo")
                
import numpy as np
from losses import MSE,MAE
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


class ML_SimpleLinearRegression(object):
    
        
    def fit(self,X_train,Y_train,showfig=False):
        
        
        self.X=X_train.astype(dtype=np.float32)
        self.Y=Y_train
        
        self.X=self.X.reshape(len(self.X),1)
        bias=np.ones(len(self.X)).reshape(len(self.X),1)
        
        self.X=np.concatenate((self.X,bias),axis=1)
        self.Weights=np.random.rand(self.X.ndim + 1)
        
        self.Weights=np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T.dot(self.Y))
        
        Y_pred=self.X.dot(self.Weights)
        
        loss=MSE(self.Y,Y_pred)        
        print("MSE loss : ",loss)
        
        
        if(showfig):
            
            plt.scatter(self.X[:,:-1],self.Y,c='b')
            plt.plot(self.X[:,:-1],Y_pred,c='r')
            plt.show()


    def predict(self,Xtest):
        
        if(Xtest.ndim==1):
            Xtest=Xtest.reshape(len(Xtest),1)
        
        bias=np.ones(len(Xtest)).reshape(len(Xtest),1)
        Xtest=np.concatenate((Xtest,bias),axis=1)
        Y_pred=Xtest.dot(self.Weights)
        
        return Y_pred
    
    
    def evaluate(self,X,Y):
        
        if(X.ndim==1):
            X=X.reshape(len(X),1)
        
        bias=np.ones(len(X)).reshape(len(X),1)
        X=np.concatenate((X,bias),axis=1)
        Y_pred=X.dot(self.Weights)
        
        loss=MSE(Y,Y_pred)
        
        print("Test MSE loss : ",loss)









class ML_MultipleLinearRegression(object):
    
        
    def fit(self,X_train,Y_train,showfig=False):
        
        
        self.X=X_train.astype(dtype=np.float32)
        self.Y=Y_train
    
        bias=np.ones(len(self.X)).reshape(len(self.X),1)
        #print(self.X.shape)
        self.X=np.concatenate((self.X,bias),axis=1)
        self.Weights=np.random.rand(self.X.ndim + 1)
        #print(self.X.shape)
        
        Y_pred=self.X.dot(self.Weights)
        loss=MAE(self.Y,Y_pred)
        #Update Weights
        
        
        self.Weights=np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T.dot(self.Y))
        print("MAE loss : ",loss)
        
        if(showfig):
            if(X_train.shape[1]==2):
                
                fig=plt.figure()
                ax=fig.add_subplot(111,projection='3d')
    
                ax.scatter3D(self.X[:,0],self.X[:,1],self.Y,c='b')
                ax.plot3D(self.X[:,0],self.X[:,1],Y_pred,c='r')
                plt.show()
                
            else:
                print("Data visualization of dimensions greater than 2 is not supported")

    def predict(self,Xtest):
        
       
        bias=np.ones(len(Xtest)).reshape(len(Xtest),1)
        Xtest=np.concatenate((Xtest,bias),axis=1)
        Y_pred=Xtest.dot(self.Weights)
        
        return Y_pred
    
    
    def evaluate(self,X,Y):
        
        bias=np.ones(len(X)).reshape(len(X),1)
        X=np.concatenate((X,bias),axis=1)
        Y_pred=X.dot(self.Weights)
        
        loss=MAE(Y,Y_pred)
        
        print("Test MAE loss : ",loss)
        
        
        
        
        
        
    
'''
##########################################
X=np.zeros((10000,2))
X[:,0]=np.array(range(10000))
X[:,1]=X[:,0]*2

Y=(X[:,0]+X[:,1])/2

X_train=X[:-1000]
Y_train=Y[:-1000]

X_test=X[-1000:]
Y_test=Y[-1000:]


model=ML_LinearRegression_ND()

model.fit(X_train,Y_train,showfig=True)

model.evaluate(X_test,Y_test)

t=model.predict(np.array([[12,24]]))
print(t)

###########
'''