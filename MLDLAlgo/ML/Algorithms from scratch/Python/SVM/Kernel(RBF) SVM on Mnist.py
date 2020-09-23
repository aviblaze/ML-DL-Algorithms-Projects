# -*- coding: utf-8 -*-
"""
Created on Sat May  9 15:35:49 2020

@author: Avinash
"""


from models import KernelSVM
from utils import get_binary_mnist

mnist_data=get_binary_mnist()

X=mnist_data[:,1:]
Y=mnist_data[:,0]

X_train=X[0:int(X.shape[0]*0.9)]
Y_train=Y[0:int(X.shape[0]*0.9)]

X_test=X[int(X.shape[0]*0.9):]
Y_test=Y[int(X.shape[0]*0.9):]

X_train/=255.0

model=KernelSVM()

model.fit(X_train,Y_train,epochs=10,C=1,learning_rate=0.0001,gamma=1,show_fig=True)

X_test/=255.0
test_acc=model.evaluate(X_test,Y_test)

print("Test Accuracy : ",test_acc)

'''
epoch :  100 Training Accuracy :  0.8124867724867725 loss :  40.25522020455311
Test Accuracy :  0.8107142857142857
'''
