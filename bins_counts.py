# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:26:39 2020

@author: 佘建友
"""
import pandas as pd
from sklearn import linear_model

filename = r"E:\MyMySql\feature\data\train_subset.csv"

#f = open(filename)

df = pd.read_csv(filename)
df.head()




# 将数据框中的分类变量转换为one-hot编码
def one_hot_df(df,column):
    one_hot_df = pd.get_dummies(df,prefix=column)
    return one_hot_df

one_hot = one_hot_df(df,"site_category")
    
    
   

     
       
























 
        
        
        
        
        
        
        
        
        
        
        
        
        

