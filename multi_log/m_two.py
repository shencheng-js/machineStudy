# -*- coding: utf-8 -*-
# @Time : 2020-11-28 21:43 
# @Author : shen
# @File : m_two.py 
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize

#这个是训练完的，通过前向传播进行计算


def load_weight(path):
    data = loadmat(path)
    return data['Theta1'], data['Theta2']

def load_data(path):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    return X,y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

theta1,theta2 = load_weight('ex3weights.mat')

print(theta1.shape,theta2.shape)

X, y = load_data('ex3data1.mat')
y = y.flatten()
# print(np.ones(X.shape[0]))
X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)  # intercept
# print(X.shape)
a1 = X

z2 = a1 @ theta1.T
#插入偏置单元
z2 = np.insert(z2, 0, 1, axis=1)
a2 = sigmoid(z2)

z3 = a2 @ theta2.T
a3 = sigmoid(z3)

#从10个反馈的值，取max
y_pred = np.argmax(a3, axis=1) + 1
accuracy = np.mean(y_pred == y)
print ('accuracy = {0}%'.format(accuracy * 100))  # accuracy = 97.52%
