# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:56:18 2020

@author: MY PC
"""

import numpy as np
import matplotlib.pyplot as plt

def Accuracy(y,y_pred):
    
    return np.mean(y==y_pred)

def sigmoid(x,w,b):
    
    return 1/(1+np.exp(-((x.dot(w))+b)))
    
def softmax(y):
    
    return np.exp(y)/np.sum(np.exp(y),axis=1).reshape(y.shape[0],1)

def One_Hot_Encode(y):
    
    assert(0 in y)
    encode=np.zeros((len(y),y.max()+1))
    for i in range(len(encode)):
        encode[i,y[i]]=1
    
    return encode
    
def Cross_Entropy(y,y_pred):
    
    y=One_Hot_Encode(y)
    assert(y.shape[1]==y_pred.shape[1])
    tot_sum=y*np.log(y_pred)
    return -tot_sum.sum()
    
def one_layer_forward_prop(x,w,b,w1,b1):
    
    hidden_layer=sigmoid(x,w,b)
    layer_out=hidden_layer.dot(w1)+b1
    
    return softmax(layer_out),hidden_layer

def one_layer_backward_prop(x,y,y_pred,w,b,w1,b1,hidden,lr):
    
    
    y=One_Hot_Encode(y)
    d1=hidden.T.dot(y-y_pred)
    W1=w1+lr*d1
    
    b1=b1+lr*((y-y_pred).sum(axis=0))
    
    d=((y-y_pred).dot(w1.T))*hidden*(1-hidden)
    d0=x.T.dot(d)
    W0=w+lr*d0
    
    b0=b+lr*(d.sum(axis=0))
    
    return W0,b0,W1,b1

x1=np.random.randn(500,2)+np.array([0,-2])    
x2=np.random.randn(500,2)+np.array([2,2])
x3=np.random.randn(500,2)+np.array([-2,2])


x=np.vstack([x1,x2,x3])
y=np.array([0]*500+[1]*500+[2]*500)

# plt.scatter(x[:,0],x[:,1],c=y)
# plt.show()

units=10 #number of units in a layer
W=np.random.rand(x.shape[1],units)
b=np.zeros(units)
   
W1=np.random.rand(units,len(np.unique(y)))
b1=np.zeros(len(np.unique(y)))

learning_rate=10e-7
epochs=1000
loss=[]

for i in range(epochs):
    forward_out,hidden=one_layer_forward_prop(x,W,b,W1,b1)
    
    if i%100==0:
        y_pred=np.argmax(forward_out,axis=1)
        cost=Cross_Entropy(y, forward_out)
        print('Accuracy : ',Accuracy(y,y_pred),'Loss : ',cost)
        loss.append(cost)
    
    W,b,W1,b1=one_layer_backward_prop(x,y,forward_out,W,b,W1,b1,hidden,learning_rate)
    
    plt.plot(loss)
    plt.show()
