# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:40:57 2020

@author: 23909
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

df = pd.read_csv(r"E:\MyMySql\PythonForDataAnalysis-master\ch08\Haiti.csv")

col_info =df.info()


np.sum(df["LATITUDE"].isnull())
np.sum(df["LONGITUDE"].isnull())
np.sum(df["APPROVED"].isnull())
np.sum(df["VERIFIED"].isnull())


df["LATITUDE"].describe()
df["LONGITUDE"].describe()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.violinplot(df["LONGITUDE"].values)


def show_distribution(Series,**kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.violinplot(Series.values)
    return plt.show()

new_df = df[["LATITUDE","LONGITUDE"]]
new_df[df["LONGITUDE"] == df["LONGITUDE"].max()]
new_df = new_df.drop(labels=[2],axis=0)


#the scater plot
sns.set_style("white")
gridobj = sns.lmplot(x="LONGITUDE",y="LATITUDE",data=new_df,
                     height=7, aspect=1.6, robust=True, palette='tab10',
                     scatter_kws=dict(s=60, linewidths=.7, edgecolors='black'))
#Decorations
gridobj.set(xlim=(0.5,7.5))
plt.title("Scatterplot with line of best fit ",fontsize=20)
plt.show()


#training data and testing data
training = new_df.sample(len(new_df)-100)
x_train=training["LONGITUDE"].values
x_train = np.mat(x_train).reshape(-1,1)
y_train= training["LATITUDE"].values
y_train = np.mat(y_train).reshape(-1,1)
testing = new_df.sample(100)
x_test = testing["LONGITUDE"].values
x_test = np.mat(x_test).reshape(-1,1)
y_test = testing["LATITUDE"].values
y_test = np.mat(y_test).reshape(-1,1)

#训练集数据放入模型中
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

print(model.predict(np.mat(np.array([50.2])).reshape(-1,1)))

#使用测试数据整体对该模型进行预测
y2 = model.predict(x_test)#预测数据
fig = plt.figure(dpi=400)
ax = fig.add_subplot(111)
ax.plot(x_test,y_test)
ax.set(title="An Example Axes",
       ylabel="Y-Axis",
       xlabel="X-Axis")
ax.legend(loc="best")
ax.xaxis.set(ticks=range(-80,-60,5))
















































