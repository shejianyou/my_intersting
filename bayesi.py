# -*- coding: utf-8 -*-
"""
Created on Sat May 23 12:22:52 2020

@author: 佘建友
"""
原理 = """
设有事件A、B
朴素贝叶斯的基本公式为:
                    P(B|A) = (P(A|B)P(B))/P(A)
从机器学习的角度:
                    P(类别|特征) = (P(特征|类别)P(类别))/P(特征)

特征条件独立:
            举例说明:指邮件中不同的词出现的概率相互之间是不受影响的，即一个词的出现不会影响另一个词的出现
            
基本定义:
        设有类别集合C={y1,y2,...,yn}
        假设特征独立；特征、标签各自具有联合概率分布，则对给定的待分类样本X={x1,x2,...,xn}，
        求解X出现的条件下各个类别yi出现的概率，哪个P(yi/X)出现的概率大，就把测试样本归类到哪一类(yi)
计算过程:
        P(yi/x) = (P(X|yi)P(yi))/P(X)
        其中：
        P(X|yi)P(yi) = P(x1|yi)P(X2|yi)....P(Xn|yi)P(yi)
        
        P(yk|X) = max{P(y1|X),P(y2|X),...,P(yi|X)},i
        
"""
一步步测试 = """
https://zhuanlan.zhihu.com/p/28720393
"""
邮件下载 = """
https://github.com/Asia-Lee/Naive_Bayes/tree/master/email
"""


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings;warnings.filterwarnings(action='once')
import pandas_datareader.data as web
import datetime
from datetime import date
import random 
import math
import numpy as np
import os
from deal_text import *
large = 22; med = 16; small = 12

params = {
    'axes.titlesize':large,
    'legend.fontsize':med,
    'figure.figsize':(16,9),
    'axes.labelsize':med,
    'axes.titlesize':med,
    'xtick.labelsize':med,
    'ytick.labelsize':med,
    'figure.titlesize':large
}
#We can tell Matplotlib to automatically make room for elements in the figures that we create
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style('white')

def createVocalList(dataSet):
    '''
  Function:        建立词集(请自查词集和词袋的本质区别:
                    词袋是在词集的基础上增加了频率的纬度，词集只关注有和没有，词袋还关注有几个。)
  Description:     计算输入各集的并集
  Calls:           
  Called By:       test
  Input:           dataSet输入的数据集,及单词向量集
  Return:          返回的是合并词集的单词向量
--------------------------------------------------------------------'''
    vocalSet = set([])#创建一个空集，用来单词去重
    for data in dataSet:
        vocalSet = vocalSet|set(data)#利用set的加法求两集合的并集
    return list(vocalSet)



def setofWords2Vec(vocalList,inputset):
    '''
      Function:     setofWords2Vec
      Description:  判断的向量的单词是否在词库中是否出现，如果出现
                    则标记为1，否则标记为0
      Calls:           
      Called By:    test
      Input:
      vocabList：
                    词库，曾经出现过的单词集合，即createVocalList
      inputSet：    测试的单词向量
                     
      Return:       returnVec是一个同vocalList同大小的数据，如果inputset单词
                    出现在vocalList则将returnVec对应的位置置1，否则为0
    -------------------------------------------------------------------'''
    returnVec = [0]*len(vocalList)#假设各个单词还没出现，即全为0
    for word in inputset:
        if word in vocalList:#如果输入单词出现在词库中，则置1
            returnVec[vocalList.index(word)] = 1
        else:
            print("the word:%s is not in my Vocabulary!"%word)
    return returnVec



def bagofWords2Vec(vocalList,inputset):

    '''
      Function:     bagofWords2Vec
      Description:  统计的单词是否在词库中出现频率(词袋)
    
      Calls:           
      Called By:    test
      Input:
      vocabList：   词库，曾经出现过的单词集合，即createVocalList
                      
      inputSet：    测试的单词向量
    
      Return:       returnVec是一个同vocalList同大小的数据，对应位置
                    表示该单词出现的个数
    -------------------------------------------------------------------
'''
    returnVec = [0]*len(vocalList)
    for word in inputset:
        if word in vocalList:
            returnVec[vocalList.index(word)] += 1
    return returnVec #返回结果



def trainNB0(trainDataSet,trainLabels):
    '''
  Function:        trainNB0
  Description:     统计侮辱性email出现概率，侮辱性邮件各单词出现的概率，非
                   侮辱性email各单词出现概率
  Calls:           
  Called By:       test
  Input:           trainDataSet：参与训练的数据集，即setofWords2Vec返回数据的集合
                   trainLabels：训练数据的标签
  Return:          
                   pShame：侮辱性email
                   p0Vect：非侮辱性email中单词出现的概率
                   p1Vect：侮辱性email中单词出现的概率
----------------------------------------------------------------------------------'''
    numTrains = len(trainDataSet) #训练数据的组数
    numWords = len(trainDataSet[0])#每组训练数据的大小
    pShame = sum(trainLabels)/float(numTrains)
    #标签中1表示侮辱，0表示非侮辱，故上述统计侮辱性email出现概率
    p0Num = np.ones(numWords) #存储统计非侮辱性邮件中各单词出现频率
    p1Num = np.ones(numWords) #存储统计侮辱性邮件中各单词出现频率
    p0SumWords = 2.0#非侮辱性邮件中单词总数
    p1SumWords = 2.0#侮辱性邮件中单词总数
    
    for i in range(numTrains):
        if trainLabels[i] == 1:#如果为侮辱性email
            p1Num += trainDataSet[i]#统计非侮辱性邮件各单词
        else:
            p0Num += trainDataSet[i]#统计侮辱性邮件各单词
    p0SumWords = sum(p0Num)#计算非侮辱性邮件中单词总数
    p1SumWords = sum(p1Num)#计算侮辱性邮件中单词总数
    p0Vect = p0Num/p0SumWords#非侮辱性邮件中各单词出现的概率
    p1Vect = p1Num/p1SumWords#侮辱性邮件中各单词出现的概率
    
    return pShame,p0Vect,p1Vect        



def classifyNB(vec2Classify,p0Vect,p1Vect,pShame):
    '''
  Function:     classifyNB
  Description:  对email进行分类
  Calls:           
  Called By:    test
  Input:
  vec2Classify：要分类的数据
  pShame：      侮辱性email
  p0Vect：      非侮辱性email中单词出现的概率
  p1Vect：      侮辱性email中单词出现的概率
  Return:          
                分类的结果，1表示侮辱性email,0表示非侮辱性email
-------------------------------------------------------------------------'''
    #下溢出：当数值过小的时候，被四舍五入为0，故用log函数对小数值的范围扩大
    temp0 = vec2Classify*p0Vect
    temp1 = vec2Classify*p1Vect
    temp11 = []
    temp00 = []
    #分布求对数，因为log不能处理array，list
    for x in temp0:
        if x>0:
            temp00.append(math.log(x))
        else:
            temp00.append(0)
    for x in temp1:
        if x>0:
            temp11.append(math.log(x))
        else:
            temp11.append(0)
    p0 = sum(temp00)+math.log(1-pShame)#属于非侮辱性email概率
    p1 = sum(temp11)+math.log(pShame)#属于侮辱性email概率
    if p1>p0:#属于侮辱性email概率大于属于非侮辱性email概率
        return 1
    else:
        return 0 



def text2VecOfWords(string):
    '''
  Function:     text2VecOfWords
  Description:  从email的string中提取单词
  Calls:           
  Called By:    test
  Input:        string： email字符串
    
  Return:       单词向量                       
---------------------------------------------------------------------'''
    import re #正则达式工具
    listOfWords = re.split(r"\w*",string)
    #分割数据，器分隔符是除单词，数字外任意的字符串
    return [word.lower() for word in listOfWords if len(word)>2]



def test(filename):
    '''
  Function:        test
  Description:     将数据部分用来训练模型，部分用来测试
  Calls:           text2VecOfWords
                   createVocalLis
                   trainNB
                   classifyNB
  Called By:       main
  Input:
  Return:          
-------------------------------------------------------------------------'''
    emailList = [] #存放每个邮件的单词向量
    emailLabel = [] #存放邮件对应的标签
    c = filename
    
    for i in range(1,26):
        #读取非侮辱性邮件，并生成单词向量
        wordList = text2VecOfWords(open((filename,i),encoding="Shift_JIS").read())
        emailList.append(wordList)#将单词向量存放到emailList中
        emailLabel.append(0)#并存放对应的标签
        wordList = text2VecOfWords(open(cwd+r"email\spam\%d.txt"%i,encoding='Shift_JIS').read())
        #读取侮辱邮件，并生成单词向量
        emailList.append(wordList)#将单词向量存放到emailList中
        emailLabel.append(1)#并存放对应的标签
        
    vocabList = createVocalList(emailList)#由所有的单词向量生成词库[w1,w2,...,wn]
    trainSet = [i for i in range(50)] #产生0-49的50个数字
    testIndex = [] #存放测试数据的下标
    for i in range(10):#从[0-49]中随机选取10个数
        randIndex = int(random.uniform(0,len(trainSet)))
        testIndex.append(trainSet[randIndex])#提取对应的数据作为训练数据
        del(trainSet[randIndex])#删除trainSet对应的值，以免下次再选中
    trainDataSet = []             #存放训练数据(用于词集方法)
    trainLabels = []              #存放训练数据标签(用于词集方法)
    
    for index in trainSet:        #trainSet剩余值为训练数据的下标
        trainDataSet.append(setofWords2Vec(vocabList,emailList[index]))
        #提取训练数据
        trainLabels.append(emailLabel[index])
        #提取训练数据标签
        
    pShame,p0Vect,p1Vect = trainNB0(trainDataSet,trainLabels)#开始训练
    errorCount = 0.0                #统计测试时分类错误的数据的个数
    for index in testIndex:
        worldVec = setofWords2Vec(vocabList,emailList[index])#数据预处理
        #进行分类，如果分类不正确，错误个位数加1
        if classifyNB(worldVec,p0Vect,p1Vect,pShame) != emailLabel[index]:
            errorCount += 1
    #输出分类错误率
    print("As to set,the error rate is :",float(errorCount)/len(testIndex))

    
    
if __name__=="__main__":
    '''
  Function:       main
  Description:    运行test函数
  Calls:          test
  Called By:       
  Input:
  Return:          
--------------------------------------------------------------------------------'''
    test()












