# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 08:13:56 2020

@author: Avinash
"""
import numpy as np
import matplotlib.pyplot as plt

def Accuracy(y,y_pred):
    
    return np.mean(y==y_pred)

def sigmoid(x,w,b):
    
    return 1/(1+np.exp(-((x.dot(w))+b)))
    
def softmax(y):
    
    return np.exp(y)/np.sum(np.exp(y),axis=1).reshape(y.shape[0],1)


def one_layer_forward_prop(x,w,b,w1,b1):
    
    Layer_pred=sigmoid(x,w,b)
    layer_out=Layer_pred.dot(w1)+b1
    
    return softmax(layer_out)




x1=np.random.randn(500,2)+np.array([-2,2])
x2=np.random.randn(500,2)+np.array([0,2])
x3=np.random.randn(500,2)+np.array([-2,0])

x=np.vstack([x1,x2,x3])
y=np.array([0]*500+[1]*500+[2]*500)

plt.scatter(x[:,0],x[:,1],c=y)
plt.show()

units=10 #number of units in a layer
W=np.random.rand(x.shape[1],units)
b=np.zeros(units)
   
W1=np.random.rand(units,len(np.unique(y)))
b1=np.zeros(len(np.unique(y)))
    
y_pred=np.argmax(one_layer_forward_prop(x,W,b,W1,b1),axis=1)
print('Accuracy : ',Accuracy(y,y_pred))


