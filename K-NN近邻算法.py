# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:44:00 2020

@author: 佘建友
"""
"""Python实现k-近邻算法"""
url = """https://blog.csdn.net/niuwei22007/article/details/49703719"""
from numpy import *
import operator
import warnings;warnings.filterwarnings(action='once')



def classify(inX,dataSet,labels,k=3):
    """
    Returns:
    使用欧式距离公式构造分类算法，返回分类结果
    --------------------------------------------
    Parameters:
    inX:输入的测试样本，是一个[x,y]样式的
    dataSet:输入的训练样本集
    labels:是训练样本标签
    k:用于选择最近邻居的数目
    """
    #先讲几个概念：
    #Numpy中，数组这一类被称为ndarray
    #1.ndarray.nidm
    #指数组的维度，即数组轴(axes)的个数，且数量等于秩(rank)。
    #最常见的矩阵就是二维数组，维度为2，轴的个数为2
    #ndarray.shape
    #shape的返回值是一个元组，元组的长度就是数组的维数，即ndim。
    #而元组中每个整数分别代表数组在其相应维度(轴)上的大小。
    
    #在本分类算法中，shape返回矩阵的[行数、列数]
    #那么shape[0]获取数据集的行数，
    #行数就是样本的数量
    dataSetSize = dataSet.shape[0]
    
    """
    求根距离过程按欧式距离的公式计算
    """
    #numpy.tile(A,reps)返回一个shape=reps的矩阵，矩阵的每个元素是A
    #一般通过向shape对应的元组添加1完成对A维度的扩充。
    #扩充完成后，则可根据reps的值对A中相对应维度的值进行重复
    #比如，A=[0,1,2],reps=2,则在第一个维度扩充1，然后第一个维度的大小为1*2
    #比如，A=[0,1,2],reps=(2,2)，则A的维度被扩充为(1,1)，然后1*2，1*2，则A的shape为(2,2),
    #A=[[0,1,2,0,1,2],
    #   [0,1,2,0,1,2]]
    
    
    #这个地方就是把测试样本扩展为和dataSet的shape一样，然后可用矩阵减法了
    diffMat=tile(inX,(dataSetSize,1))-dataSet#diffMat就是输入样本与每个训练样本的差值
    sqDiffMat = diffMat**2#平方
    sqDistances=sqDiffMat.sum(axis=1)#按横轴相加，返回一个一维数组
    distances=sqDistances**0.5#取平方根
    sortedDistIndicies=distances.argsort()#按照距离递增次序排序，原下标
    #比如，x=[30,10,20,40]
    #那么升序排序后是[10,20,30,40]，他们的原下标是[1,2,0,3]
    #那么，numpy.argsort(x)=[1,2,0,3]
    #存放最终分类结果即相应的结果投票数
    classCount={}
    #投票过程，就是统计前k个最近的样本所属类别包含的样本个数
    for i in range(k):
        #index = sortedDistIndicies[i]是第i个最相近的样本下标
        #voteIlabel=labels[index]#是样本index对应的分类结果("A"or"B")
        votellabel = labels[sortedDistIndicies[i]]
        #classcount.get(votellabel,0)返回votellabel的值,
        #如果不存在,则返回0，
        #然后将票数增1
        classCount[votellabel]=classCount.get(votellabel,0)+1#dict.get(key,[default])Return the value for key if key is in the dictionary, else default.
    #把分类结果进行排序，然后返回得票数最多的分类结果
    sortedClassCount = sorted(classCount.items(),
                             key=operator.itemgetter(1),reverse=True)##if multiple items are specified, returns a tuple of lookup values
    return sortedClassCount[0][0]


def autoNorm(dataSet):
    """
    Function:
            对特征数据进行标准化
    --------------------------------------------
    Returns:
            normDataSet:标准化后的特征向量(二维的)
            ranges:特征向量的极差
            minVals:特征向量的最小值
    --------------------------------------------
    Params:
            dataSet:二维特征数据
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet-tile(minvals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    
    return normDataSet,ranges,minVals


def file2matrix(filename):
    """
    Function:
            实现从文件中读取数据。
    ----------------------------------------
    Returns:
            returnMat:特征向量(二维的)
            ckassLabelVector:标签向量(一维的)
    ----------------------------------------
    Params:
            filename:文件地址
    """
    fr = open(filename)
    numberOfLines = len(fr.readlines())#to return a list of lines 
    for line in fr.readlines():
        line = line.strip().split("\t")
        cols = len(line)
        break
    returnMat = zeros((numberOfLines,cols-1))
    classLabelVector = []
    index=0
    #解析文件数据到列表
    for line in fr.readlines():
        line = line.strip()#to copy line after omiiting
        ListFormLine = line.split("\t")#to return a list of words
        returnMat[index,:] = ListFormLine[0:cols-1]
        classLabelVector.append((ListFormLine[-1]))
        index+=1
        
    return returnMat,classLabelVector


def datingClassTest():
    """
    Function:
            测试约会网站分类器性能
    -----------------------------------------------
    Returns:
            分类器(classify)返回的标签
            正确的标签
            总的错误率
            错误个数
    -----------------------------------------------
    Params:
           None
    """
    hoRatio = 0.05#取多少行特征向量用来测试
    datingDataMat,datingLabels = file2matrix(filename)
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i,:],
        normMat[numTestVecs:m,:],datingLabels[numTestVecs:m])
        print("分类器返回的值:%s,正确的值:%s"
             %(classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("总的错误率是：%f"%(errorCount/float(numTestVecs)))
    print("错误的个数:%f"%errorCount)
    

    
def classifyPerson():
    """
    
    """
    resultList = ["不喜欢","一般喜欢","特别喜欢"]
    percentTats = float(input("玩游戏占的百分比"))
    ffMiles=float(input("每年坐飞机多少公里"))
    iceCream = float(input("每年吃多少公升的冰淇淋"))
    datingDataMat,datingLabels = file2matrix(filename)
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify((inArr-minVals)/ranges,normMat,datingLabels)
    print("你将有可能对这个人是:",resulList[int(classifierResult)-1])



