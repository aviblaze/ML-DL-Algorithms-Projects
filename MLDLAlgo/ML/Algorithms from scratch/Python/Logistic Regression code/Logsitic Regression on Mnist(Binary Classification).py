# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 15:30:05 2020

@author: Avinash
"""


from models import ML_SimpleLogisticRegression
from utils import get_binary_mnist

mnist_data=get_binary_mnist()

X=mnist_data[:,1:]
Y=mnist_data[:,0]

X_train=X[0:int(X.shape[0]*0.9)]
Y_train=Y[0:int(X.shape[0]*0.9)]

X_test=X[int(X.shape[0]*0.9):]
Y_test=Y[int(X.shape[0]*0.9):]

X_train/=255.0

model=ML_SimpleLogisticRegression()

model.fit(X_train,Y_train,epochs=2000,showfig=True,learning_rate=0.001)

X_test/=255.0
test_acc=model.evaluate(X_test,Y_test)

print("Test Accuracy : ",test_acc)

'''
epoch :  2000 loss :  0.2646647578695903 Accuracy :  0.9047138895891101
Test Accuracy :  0.9013605442176871
'''