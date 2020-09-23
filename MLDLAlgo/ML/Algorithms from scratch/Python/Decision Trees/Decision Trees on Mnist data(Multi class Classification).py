# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:51:19 2020

@author: Avinash
"""
from models import DecisionTreeClassifier
from utils import get_mnist


mnist_data=get_mnist()

X=mnist_data[:,1:]
Y=mnist_data[:,0]

X_train=X[0:int(X.shape[0]*0.9)]
Y_train=Y[0:int(X.shape[0]*0.9)]

X_test=X[int(X.shape[0]*0.9):]
Y_test=Y[int(X.shape[0]*0.9):]

X_train/=255.0

model=DecisionTreeClassifier()

model.fit(X_train,Y_train,min_split=10,depth=15)
print("Training completed")

X_test/=255.0

test_acc=model.evaluate(X_test,Y_test)

print("Test Accuracy : ",test_acc)

'''
Depth->5
'''