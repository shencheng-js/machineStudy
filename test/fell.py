# -*- coding: utf-8 -*-
# @Time : 2020-11-25 10:17 
# @Author : shen
# @File : fell.py 
# @Software: PyCharm

import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

#读入数据并对其录取结果修改为 -1，1
path = r'./ex2data1.txt'
data = pd.read_csv(path,names=['exam1','exam2','admitted'])
# print(data)
# data[data.admitted==0].admitted = -1
data.replace(0,-1,inplace=True)
data.insert(0,"one",1)

X=data.iloc[:,:-1].values
y = data.iloc[:,-1].values
# print(X)

theta = np.array([1,1,1])

alpha = 0.01
epoch = 1000


def finderror(X,y,theta):
    result = X@theta
    # print(result)
    lens = len(y)
    for i in range(lens):

        if result[i]*y[i] < 0:
            return i

    return -1
def felling(X,y,theta,alpha,epoch):
    lens = len(y)
    for i in range(epoch):
        sit,re = finderror(X,y,theta)
        if re==-1:
            break
        else:
            theta = theta+alpha*y[re]*X[re]

    return theta


theta = felling(X,y,theta,0.01,100000)

result = X@theta
test = 0
lens = len(result)
for i in range(lens):
    if result[i]*y[i]<0:
        test+=1
    print(str(result[i])+","+str(y[i]))

print(1.0*test/lens)