# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:24:31 2020

@author: Avinash

"""
import sys
sys.path.append("E:\\MLDLAlgo")
                
import re
import numpy as np
from utils import find_boundaries,get_nested_dict_value
from metrics import Entropy,Accuracy

sys.path.append("E:\\MLDLAlgo")


class DecisionTreeClassifier(object):
    
    
    def find_best_feature(self,x,y):
    
        before_ig=Entropy(y)
        best_val=0
        best_feat=0
        best_feat_ig=0
        if(before_ig==0):
            return [-1,-1,0]
        
        for i in range(x.shape[1]):
            
            
            tmp_x=x[:,i]
            
            if(re.match("<U+",tmp_x.dtype.descr[0][1])):
                
                indiv_entr=0
                best_ig=0
                for j in set(tmp_x):
                    
                    tmp_y1=y[tmp_x==j]
                    tmp_y2=y[tmp_x!=j]                    
                    entr1=Entropy(tmp_y1)
                    entr2=Entropy(tmp_y2)
                    ig_gain=before_ig-((len(tmp_y1)/len(y))*entr1+(len(tmp_y2)/len(y))*entr2)
                    if(ig_gain>best_ig): 
                        best_val=j
                        best_feat=i
                        best_ig=ig_gain
                    
                    indiv_entr-=(tmp_y1/len(y))*entr1+(tmp_y2/len(y))*entr2
                
                if(before_ig-indiv_entr>best_feat_ig):
                    best_val=j
                    best_feat=i
                    best_feat_ig=before_ig-indiv_entr
                    
                    
            else:
                
                boundaries=find_boundaries(tmp_x,y)
               
                for j in boundaries:
                    
                    tmp_y1=y[tmp_x<=j]
                    tmp_y2=y[tmp_x>j]                    
                    entr1=Entropy(tmp_y1)
                    entr2=Entropy(tmp_y2)
    
                    ig_gain=before_ig-((len(tmp_y1)/len(y))*entr1+(len(tmp_y2)/len(y))*entr2)
                    
                    if(ig_gain>best_feat_ig): 
                        best_val=j
                        best_feat=i
                        best_feat_ig=ig_gain
                                 
        return [best_feat,best_val,best_feat_ig]
        
            
    def create_node(self,X,Y,feat,val,side,cur_depth):
        
        
        if(cur_depth<self.Depth and len(X)>self.Min_split):
            
            best_feature=self.find_best_feature(X,Y)
            
            if(feat==None):
                if(best_feature[2]==0):
                    self.Tree[(best_feature[0],best_feature[1])]=Y.argmax()
                    
                self.Tree[(best_feature[0],best_feature[1])]={'left':None,'right':None}
                
            else:           
                if(side=='left'):
                    if(best_feature[2]==0):
                        node=get_nested_dict_value((feat,val),self.Tree,None)
                        node['left']=Y.argmax()  
                        return
                    else:
                        node=get_nested_dict_value((feat,val),self.Tree,None)
                        node['left']={(best_feature[0],best_feature[1]):{'left':None,'right':None}}     
                else:
                    if(best_feature[2]==0):
                        node=get_nested_dict_value((feat,val),self.Tree,None)
                        node['right']=Y.argmax()  
                        return
                    else:
                        node=get_nested_dict_value((feat,val),self.Tree,None)
                        node['right']={(best_feature[0],best_feature[1]):{'left':None,'right':None}}
        
            cur_depth+=1 
            best_feat_dtype=X[:,best_feature[0]].dtype.descr[0][1]
            
            if(re.match("<U+",best_feat_dtype)):
                left_x=X[X[:,best_feature[0]]==best_feature[1]]
                left_y=Y[X[:,best_feature[0]]==best_feature[1]]
                
                right_x=X[X[:,best_feature[0]]!=best_feature[1]]
                right_y=Y[X[:,best_feature[0]]!=best_feature[1]]
                
               
               
                self.create_node(left_x, left_y, best_feature[0], best_feature[1], 'left',cur_depth)  
                self.create_node(right_x, right_y, best_feature[0], best_feature[1], 'right',cur_depth)
                
                    
            else:
                left_x=X[X[:,best_feature[0]]<=best_feature[1]]
                left_y=Y[X[:,best_feature[0]]<=best_feature[1]]
                
                right_x=X[X[:,best_feature[0]]>best_feature[1]]
                right_y=Y[X[:,best_feature[0]]>best_feature[1]]
                
                print("Depth : ",cur_depth)
                print(left_x.shape,right_x.shape)
                self.create_node(left_x, left_y, best_feature[0], best_feature[1], 'left',cur_depth)  
                self.create_node(right_x, right_y, best_feature[0], best_feature[1], 'right',cur_depth)
                
                
        else:
            
            if(side=='left'):
                node=get_nested_dict_value((feat,val),self.Tree,None)     
                node['left']=np.bincount(Y).argmax()
                
                
            elif(side=='right'):
                node=get_nested_dict_value((feat,val),self.Tree,None)     
                node['right']=np.bincount(Y).argmax()
                
            return
            
    def fit(self,x,y,min_split=2,depth=2):
        
        self.X=x
        self.Y=y.astype(np.int32)
        self.Depth=depth
        self.Min_split=min_split
        
        self.Tree={}
        Cur_depth=-1
        
        self.create_node(self.X, self.Y, None, None, None,Cur_depth)
        
        
    def predict(self,x):
        
        Y_pred=np.zeros(len(x))
        
        for i in range(len(x)):
            
            use_dictionary=self.Tree
            node=list(use_dictionary.keys())[0]
            
            while(type(node) is tuple):
                
                ind,val=node
                if(re.match("<U+",x[i][ind].dtype.descr[0][1])):
                        
                    if(x[i][ind]==val):
                        if(type(use_dictionary[( ind,val)]['left']) is dict):
                            use_dictionary=use_dictionary[(ind,val)]['left']
                            node=list(use_dictionary.keys())[0]
                            
                        else:
                            Y_pred[i]=use_dictionary[( ind,val)]['left']
                            break
                        
                    else:
                        if(type(use_dictionary[( ind,val)]['right']) is dict):
                            use_dictionary=use_dictionary[(ind,val)]['right']
                            node=list(use_dictionary.keys())[0]
                            
                        else:
                            Y_pred[i]=use_dictionary[( ind,val)]['right']
                            break
                else:
                    if(x[i][ind]<=val):
                        if(type(use_dictionary[( ind,val)]['left']) is dict):
                            use_dictionary=use_dictionary[(ind,val)]['left']
                            node=list(use_dictionary.keys())[0]
                            
                        else:
                            Y_pred[i]=use_dictionary[( ind,val)]['left']
                            break
                    else:
                        #print("====================================================")
                        #print(use_dictionary)
                        if(type(use_dictionary[( ind,val)]['right']) is dict):
                            use_dictionary=use_dictionary[(ind,val)]['right']
                            node=list(use_dictionary.keys())[0]
                            
                        else:
                            Y_pred[i]=use_dictionary[( ind,val)]['right']
                            break
            else:
                Y_pred[i]=node
            
            
               
        return Y_pred

    def evaluate(self,x,y):
    
        Y_pred=self.predict(x)
        
        return Accuracy(y,Y_pred)

    
    
    
    
    
    
    
    
    
    
################################################################################