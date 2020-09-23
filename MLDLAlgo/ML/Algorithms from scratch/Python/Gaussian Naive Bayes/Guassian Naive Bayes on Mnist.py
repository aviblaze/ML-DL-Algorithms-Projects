# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:28:29 2020

@author: Avinash
"""
from models import Gaussian_NaiveBayes
from utils import get_mnist

mnist_data=get_mnist()

X=mnist_data[:,1:]
Y=mnist_data[:,0]


X_train=X[0:int(X.shape[0]*0.9)]
Y_train=Y[0:int(X.shape[0]*0.9)]

X_test=X[int(X.shape[0]*0.9):]
Y_test=Y[int(X.shape[0]*0.9):]

X_train/=255.0

model=Gaussian_NaiveBayes()

model.fit(X_train,Y_train)

train_acc=model.evaluate(X_train,Y_train)
print("Train Accuracy : ",train_acc)


X_test/=255.0
test_acc=model.evaluate(X_test,Y_test)

print("Test Accuracy : ",test_acc)

'''
Train Accuracy :  0.8016137566137567
Test Accuracy :  0.7966666666666666
'''
