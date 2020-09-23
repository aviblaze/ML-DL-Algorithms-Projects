# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:32:44 2020

@author: Avinash
"""

from models import DecisionTreeClassifier
from utils import get_binary_mnist


mnist_data=get_binary_mnist()

X=mnist_data[:,1:]
Y=mnist_data[:,0]

X_train=X[0:int(X.shape[0]*0.9)]
Y_train=Y[0:int(X.shape[0]*0.9)]

X_test=X[int(X.shape[0]*0.9):]
Y_test=Y[int(X.shape[0]*0.9):]

X_train/=255.0

model=DecisionTreeClassifier()

model.fit(X_train,Y_train,min_split=10,depth=5)
print("Training completed")

X_test/=255.0

test_acc=model.evaluate(X_test,Y_test)

print("Test Accuracy : ",test_acc)

'''
Depth->2

Depth :  0
(3648, 784) (4286, 784)
Depth :  1
(3635, 784) (13, 784)
Depth :  2
(3615, 784) (20, 784)
Depth :  2
(11, 784) (2, 784)
Depth :  1
(4204, 784) (82, 784)
Depth :  2
(4192, 784) (12, 784)
Depth :  2
(25, 784) (57, 784)

Training completed
Test Accuracy :  0.9909297052154195

Depth->5

Depth :  0
(3648, 784) (4286, 784)
Depth :  1
(3635, 784) (13, 784)
Depth :  2
(3615, 784) (20, 784)
Depth :  3
(3523, 784) (92, 784)
Depth :  4
(3522, 784) (1, 784)
Depth :  4
(11, 784) (81, 784)
Depth :  5
(4, 784) (7, 784)
Depth :  3
(9, 784) (11, 784)
Depth :  2
(11, 784) (2, 784)
Depth :  1
(4204, 784) (82, 784)
Depth :  2
(4192, 784) (12, 784)
Depth :  3
(4191, 784) (1, 784)
Depth :  3
(7, 784) (5, 784)
Depth :  2
(25, 784) (57, 784)
Depth :  3
(17, 784) (8, 784)
Depth :  4
(16, 784) (1, 784)

Training completed
Test Accuracy :  0.4977324263038549
'''