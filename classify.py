# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 20:14:43 2020

@author: 23909
"""
"""
进行文本无监督聚类操作
1.语料加载
2.分词
3.去停用词
4.抽取词向量特征
5.实战TF-IDF的中文文本K-means聚类
6.实战word2Vec的中文文本K-means聚类
7.结果可视化
"""
import matplotlib.pyplot as plt
import pandas as pd
def k_means(train_tfidf,n_clusters=10):
        """
        Parameters:
                train_tfidf:train set;
                n_clusters:the number of clusters;
                n_init:Number of time the k-means algorithm will be run with different centroid seeds;
                max_iter:Maximum number of iterations of the k-means algorithm for a single sun
                n_jobs:The number fo OpenMP threads to use for the computation.
                algorithm:{"suto","full",elkan},default="auto"
            
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.feature_extraction.text import TfidfVectorizer
          
        kmeans = KMeans(n_clusters=n_clusters,random_state=0).fit(train_tfidf)
        #lebels = kmeans.labels_
        results = kmeans.predict(train_tfidf)
        centers = kmeans.cluster_centers_
        score = ("%.2f"%kmeans.score(train_tfidf)).replace("-"," ")
        #score:Opposite of the value of the train set on the K-means objective 
        return results,centers,score
    

#第二步,定义聚类结果可视化函数
#plot_cluster(result,newData,numClass)
def plot_cluster(result,newData,numClass):
    """
    Parameters:
        result表示聚类拟合的结果集
        newData表示权重weight降维的结果,
        numClass表示聚类分为几簇
    Returns:
        第一部分绘制结果newData
        第二部分绘制聚类的中心点
    """
    plt.figure(2)
    Lab = [[] for i in range(numClass)]
    index = 0
    for labi in result:
        Lab[labi].append(index)
        index+=1
    color = ["oy","ob","og","cs","ms","bs","ks",
             "g^"
        ]*3
    for i in range(numClass):
        x1 = []
        y1 = []
        for ind1 in newData[Lab[i]]:
            try:
                y1.append(ind1[1])
                x1.append(ind1[0])
            except:
                pass
        plt.plot(x1,y1,color[i])
    
    #绘制初始中心点
    x1 = []
    y1 = []
    for ind1 in clf.cluster_centers_:
        try:
            y1.append(ind1[1])
            x1.append(ind1[0])
        except:
            pass
    plt.plot(x1,y1,"rv")#绘制中心
    plt.show()


"""
进行有监督的文本分类
1.语料加载
2.分词
3.去停用词
4.抽取词向量特征
5.分别进行算法建模和模型训练
6.评估、计算AUC值
7.模型对比
"""

from sklearn.naive_bayes import MultinomialNB,GaussianNB
classifier = MultinomialNB()
classifier.fit(x_train,y_train)
print(classifier.score(x_test,y_test))

#TypeError:dense data is required.
#Use X.toarray() and PCA_ functions to convert to a dense numpy array
gNB = GaussianNB()
gNB.fit(x_train,y_train)
print(gNB.score(x_test,y_test))

#改变训练模型，在不改变特征向量模型的前提下
from sklearn.svm import SVC
svm = SVC(kernel="linear")
svm.fit(x_train,y_train)
print(svm.score(x_test,y_test))




