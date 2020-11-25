# -*- coding: utf-8 -*-
# @Time : 2020-11-23 10:21 
# @Author : shen
# @File : one.py 
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

path = r'D:\fireFox_download\机器学习题目代码及数据文件\线性回归\数据文件\ex1data1.txt'
data = pd.read_csv(path,header=None,names=['Population', 'Profit'])
# print(data.head())


#经过测试输入数据正常

#插入数据,作为X0的值
data.insert(0,"one",1)

mycost = []
# print(data.head())
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

theta = np.zeros(X.shape[1])
# print(theta)

#计算代价，这个和梯度下降是两回事，这个是损失，而梯度下降是求最佳参数
def computecost(X,y,theta):
    inner = np.power((X@theta-y),2)
    return inner.sum()/(2*len(y))

#计算梯度下降,alpha是学习率，epoch是次数
def gradientDescent(X, y, theta, alpha, epoch):
    leny = len(y)
    for i in range(epoch):
        theta = theta - (alpha/leny)*(X.T@(X@theta-y))
        mycost.append(computecost(X,y,theta))

    return theta

alpha = 0.01
epoch = 10000
theta = gradientDescent(X,y,theta,alpha,epoch)
print(theta)
print(mycost[-1])
# 基本操作结束，绘图即可

# fig=plt.figure(num=1,figsize=(4,4))
# ax = fig.add_su
x = np.linspace(data.Population.min(), data.Population.max(), 1000)
y=theta[0]+theta[1]*x
# ax.plot(x,y,'r',label='Prediction')
# plt.show()
fig=plt.figure(num=1,figsize=(10,10))
ax=fig.add_subplot(111)
ax.set_xlim(0,25)
ax.plot(x,y,"r-.d",label="苹果",linewidth=1.0)

ax.scatter(data['Population'],data['Profit'])
ax.set_xlabel("Population")
ax.set_ylabel("Profit")
ax.set_title("predict")
plt.show()

# model = linear_model.LinearRegression()
# machine=model.fit(X, y)
# x = np.array(X[:, 1])
# f = model.predict(X).flatten()
#
# fig, ax = plt.subplots(figsize=(8,5))
# ax.plot(x, f, 'r', label='Prediction')
# ax.scatter(data.Population, data.Profit, label='Traning Data')
# ax.legend(loc=2)
# ax.set_xlabel('Population')
# ax.set_ylabel('Profit')
# ax.set_title('Predicted Profit vs. Population Size')
# plt.show()