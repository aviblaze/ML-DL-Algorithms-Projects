# -*- coding: utf-8 -*-
"""
Created on Fri May  1 12:24:40 2020

@author: Avinash
"""

from models import simple_LinearSVM
from utils import get_binary_mnist

mnist_data=get_binary_mnist()

X=mnist_data[:,1:]
Y=mnist_data[:,0]

X_train=X[0:int(X.shape[0]*0.9)]
Y_train=Y[0:int(X.shape[0]*0.9)]

X_test=X[int(X.shape[0]*0.9):]
Y_test=Y[int(X.shape[0]*0.9):]

X_train/=255.0

model=simple_LinearSVM()

model.fit(X_train,Y_train,epochs=400,C=0.001,learning_rate=0.001,show_fig=True)

X_test/=255.0
test_acc=model.evaluate(X_test,Y_test)

print("Test Accuracy : ",test_acc)

'''
epoch :  397 Training Accuracy :  0.9653390471388958
epoch :  398 Training Accuracy :  0.9654650869674817
epoch :  399 Training Accuracy :  0.9654650869674817
epoch :  400 Training Accuracy :  0.9654650869674817
Test Accuracy :  0.9727891156462585
'''