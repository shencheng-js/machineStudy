# -*- coding: utf-8 -*-
# @Time : 2020-11-25 11:42 
# @Author : shen
# @File : feel2.py 
# @Software: PyCharm

import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

# 读入数据并对其录取结果修改为 -1，1
path = r'./ex2data1.txt'
data = pd.read_csv(path, names=['exam1', 'exam2', 'admitted'])
# print(data)
data.replace(0, -1, inplace=True)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
b = 1
theta = np.array([1, 1])


def act(x):
    if x >= 0:
        return 1
    elif x < 0:
        return -1


def feeling(X, y, theta, alpha, epoch, b):
    lens = len(y)
    for i in range(epoch):
        for j in range(lens):
            nowx = X[j]
            result = nowx @ theta + b
            temp = act(result - y[j])
            if temp == 0:
                continue
            elif temp < 0:
                theta = theta - alpha * nowx
                b = b - alpha*temp
            else:
                theta = theta + alpha * nowx
                b = b + alpha*temp

    return theta


theta = feeling(X, y, theta, 0.001, 2, b)
print(theta)

result = X@theta +b
test = 0
lens = len(result)
for i in range(lens):
    if result[i]*y[i]<0:
        test+=1
    print(str(result[i])+","+str(y[i]))
