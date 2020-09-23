# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:16:04 2020

@author: Avinash
"""
import numpy as np
import re

def sigmoid(X,W,b):
    
    return 1/(1+np.exp(-(X.dot(W)+b)))

########################################################################################################3

def softmax(X,W,b):
    
    Y_pred=np.exp(X.dot(W)+b)
    
    return Y_pred/np.sum(Y_pred,axis=1).reshape(len(Y_pred),1)
    
########################################################################################################3    

def get_binary_mnist():
    
    mnist_data=np.zeros((42000,785))
    with open('E:\\AI and ML\\Mnist Dataset\\train.csv') as data:
        
        
        c=-1
        for _line in data:
            if(c==-1):
                c+=1
                continue
            _line=re.sub('\n','',_line).split(',')
            mnist_data[c,:]=[int(i) for i in _line]
            c+=1
    
    class_one=mnist_data[:,0]==1
    class_zero=mnist_data[:,0]==0
    data=mnist_data[np.logical_or(class_one,class_zero)]
    
    return data

########################################################################################################3

def get_mnist():
    
    mnist_data=np.zeros((42000,785))
    with open('E:\\AI and ML\\Mnist Dataset\\train.csv') as data:
        
        
        c=-1
        for _line in data:
            if(c==-1):
                c+=1
                continue
            _line=re.sub('\n','',_line).split(',')
            mnist_data[c,:]=[int(i) for i in _line]
            c+=1
            
    
    return mnist_data

########################################################################################################3

def One_Hot_Encode(data):
    
    data=data.astype(np.int32)
    transformed=np.zeros((len(data),int(max(set(data))+1)))
    transformed[range(len(data)),data]=1
    
    return transformed




########################################################################################################

def find_boundaries(xvalues,labels):
    
    sort_ind=xvalues.argsort()
    xvalues=xvalues[sort_ind]
    labels=labels[sort_ind]
    
    label=set(labels)
    boundaries=[]
    boundary_values=[]
    for i in label:
        
        one_arr=np.array([i]*len(labels))
        one_arr[labels==i]=1
        one_arr[labels!=i]=0
        #print(i,one_arr)
        for j in range(len(one_arr)-1):
            
            if((one_arr[j]+one_arr[j+1]) == 1 and j not in boundaries):
                boundaries.append(j)
    
    boundaries=np.array(boundaries)
    boundary_values=(xvalues[boundaries]+xvalues[boundaries+1])/2
    
   
    return set(boundary_values)
    
    
    
    
########################################################################################################

def update_nested_dict(key,val,dictionary,status):
    
    search_key=key
    search_value=val
    update_status=status
    
    tmp=dictionary
    if(search_key in tmp):
        
        tmp[search_key]=search_value
        return 'updated'
    
    else:
        for _k,_v in tmp.items():
            if(update_status!='updated'):
                if(type(_v) is dict):
                    update_status=update_nested_dict(search_key,search_value,_v,update_status)
                else:
                    continue
            else:
                return 'updated'
            
            
########################################################################################################            

def get_nested_dict_value(key,dictionary,status):
    
    search_key=key
    update_status=status
    tmp=dictionary
    
    if(search_key in tmp):
        return tmp[search_key]
    
    else:
        for _k,_v in tmp.items():
            if(update_status==None):
                if(type(_v) is dict):
                    update_status=get_nested_dict_value(search_key,_v,update_status)
                    if(update_status!=None):
                        return update_status
                    
                else:
                    continue
            else:
                return update_status
                      
            
########################################################################################################