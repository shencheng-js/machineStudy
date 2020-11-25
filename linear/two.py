# -*- coding: utf-8 -*-
# @Time : 2020-11-23 16:15 
# @Author : shen
# @File : two.py 
# @Software: PyCharm

# 需要对数据进行归一化
# data2 = (data2 - data2.mean()) / data2.std()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

path = r'D:\fireFox_download\机器学习题目代码及数据文件\线性回归\数据文件\ex1data2.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
# print(data.head())
data = (data - data.mean()) / data.std()
# print(data.head())
data.insert(0,"one",1)

#cost列表
mycost = []
#获取对应值
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

theta = np.zeros(X.shape[1])

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
#  手动实现梯度下降
# alpha = 0.01
# epoch = 100000
# theta = gradientDescent(X,y,theta,alpha,epoch)
# print(theta)
# print(mycost[1],mycost[-1])

# model = linear_model.LinearRegression()
# machine=model.fit(X, y)
# x = np.array(X[:, 1])
# f = model.predict(X).flatten()
#
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()