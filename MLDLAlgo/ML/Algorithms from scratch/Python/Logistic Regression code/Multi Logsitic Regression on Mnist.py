# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:04:38 2020

@author: MY PC
"""

from models import ML_MultiLogisticRegression
from utils import get_mnist,One_Hot_Encode

mnist_data=get_mnist()

X=mnist_data[:,1:]
Y=mnist_data[:,0]
Y=One_Hot_Encode(Y)

X_train=X[0:int(X.shape[0]*0.9)]
Y_train=Y[0:int(X.shape[0]*0.9)]

X_test=X[int(X.shape[0]*0.9):]
Y_test=Y[int(X.shape[0]*0.9):]

X_train/=255.0

model=ML_MultiLogisticRegression()

model.fit(X_train,Y_train,epochs=800,showfig=True,learning_rate=0.1)

X_test/=255.0
test_acc=model.evaluate(X_test,Y_test)

print("Test Accuracy : ",test_acc)

'''
epoch :  800 loss :  146.13381813147961 Accuracy :  78.21957671957672
Test Accuracy :  77.97619047619048
'''