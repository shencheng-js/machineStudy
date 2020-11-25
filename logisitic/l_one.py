# -*- coding: utf-8 -*-
# @Time : 2020-11-23 16:38 
# @Author : shen
# @File : l_one.py 
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

path = r'D:\fireFox_download\机器学习题目代码及数据文件\logistics回归\数据文件\ex2data1.txt'
data = pd.read_csv(path, header=None, names=['exam1', 'exam2', 'admitted'])
# print(data.head())
# data = (data - data.mean()) / data.std()
# print(data.head())
data.insert(0,"one",1)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

theta = np.zeros(X.shape[1])

#获取被录取及不被录取的人
positive = data[data.admitted.isin(['1'])]  # 1
negetive = data[data.admitted.isin(['0'])]  # 0



#激活函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#代价函数
def cost(theta, X, y):
    first = (-y) * np.log(sigmoid(X @ theta))
    second = (1 - y)*np.log(1 - sigmoid(X @ theta))
    return np.mean(first - second)

def gradient(theta, X, y):
    theta = (X.T @ (sigmoid(X @ theta) - y))/len(X)
    return theta
    # return (X.T @ (sigmoid(X @ theta) - y))/len(X)

result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
# print(cost(result[0],X,y))
re = result[0]
x = np.arange(0,100)
y = -(re[0]+re[1]*x)/re[2]

#作图
fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(positive['exam1'], positive['exam2'], c='b', label='Admitted')
ax.scatter(negetive['exam1'], negetive['exam2'], s=50, c='r', marker='x', label='Not Admitted')
# 设置图例显示在图的上方
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])
ax.legend(loc='center left', bbox_to_anchor=(0.2, 1.12),ncol=3)
# 设置横纵坐标名
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')

ax.plot(x,y)
plt.show()

