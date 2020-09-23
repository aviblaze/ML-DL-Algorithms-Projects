# -*- coding: utf-8 -*-
"""
Created on Sat May  2 07:51:11 2020

@author: Avinash
"""

from model import KNN
from utils import get_mnist

mnist_data=get_mnist()

X=mnist_data[:,1:]
Y=mnist_data[:,0]

X_train=X[0:int(X.shape[0]*0.9)]
Y_train=Y[0:int(X.shape[0]*0.9)]

X_test=X[int(X.shape[0]*0.9):]
Y_test=Y[int(X.shape[0]*0.9):]

#X_train/=255.0

model=KNN()

model.fit(X_train,Y_train,n=5)

#X_test/=255.0
test_acc=model.evaluate(X_test,Y_test)

print("Test Accuracy : ",test_acc)

'''
Test Accuracy :  0.9671428571428572
'''