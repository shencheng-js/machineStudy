# -*- coding: utf-8 -*-
# @Time : 2020-11-25 10:57 
# @Author : shen
# @File : test.py 
# @Software: PyCharm


import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

# 读入数据并对其录取结果修改为 -1，1
path = r'./ex2data3.txt'
data = pd.read_csv(path, names=['exam1', 'exam2', 'admitted'])
# print(data)
# data.replace(0, -1, inplace=True)

positive = data[data.admitted.isin(['1'])]
negetive = data[data.admitted.isin(['-1'])]
# print(positive)

items = data.iloc[:,:].values
w = [1,1]          #初始化w参数
b = 1              #初始化b参数


def update(item,alpha):
    global w,b
    lens = len(item)
    #更新w值
    for i in range(lens-1):
        w[i]+=alpha*item[-1]*item[i]
    b += alpha*item[-1]
    # print ('w = ',w,'b=',b)                     #打印出结果

def judge(item):                               #返回y = yi(w*x+b)的结果
    res = 0
    for i in range(len(item)-1):
        res +=item[i]*w[i]                   #对应公式w*x
    res += b                                    #对应公式w*x+b
    res *= item[-1]                              #对应公式yi(w*x+b)
    return res


def check():                                    #检查所有数据点是否分对了
    flag = False
    for item in items:
        if judge(item)<=0:                       #如果还有误分类点，那么就小于等于0
            flag = True
            update(item,0.001)                         #只要有一个点分错，我就更新
    return flag                                  #flag为False，说明没有分错的了

if __name__ == "__main__":
    flag = False
    for i in range(10000):
        if not check():  # 如果已经没有分错的话
            flag = True
            break
    if flag:
        print("在1000次以内全部分对了")
    else:
        print("很不幸，1000次迭代还是没有分对")

    fig = plt.figure(num=1,figsize=(6,8))
    ax=fig.add_subplot(111)
    ax.scatter(positive['exam1'], positive['exam2'], c='b', label='Admitted')
    ax.scatter(negetive['exam1'], negetive['exam2'], s=50, c='r', marker='x', label='Not Admitted')

    print(w,b)

    x = np.linspace(data.Population.min(), data.Population.max(), 1000)
    y = theta[0] + theta[1] * x
    # ax.plot(x,y,'r',label='Prediction')
    # plt.show()
    fig = plt.figure(num=1, figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 25)
    ax.plot(x, y, "r-.d", label="划分", linewidth=1.0)
    # x = data.iloc[:,:-1].values
    # y = x*w+b
    # ax.plot(x,y)
    plt.show()