# -*- coding: utf-8 -*-
# @Time : 2020-12-3 20:19 
# @Author : shen
# @File : o_one.py 
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt


def load_mat(path):
    '''读取数据'''
    data = loadmat(path)  # return a dict
    X = data['X']
    y = data['y'].flatten()

    return X, y


def plotData(X, y):
    """瞧一瞧数据长啥样"""
    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 1:], y, c='r', marker='x')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.grid(True)
    # plt.show()


def costReg(theta, X, y, l=1):
    """正则化代价函数"""
    cost = ((X @ theta - y.flatten()) ** 2).sum()
    reg = l * (theta[1:] @ theta[1:])
    return (cost + reg) / (2 * len(X))


def gradientReg(theta, X, y, l):
    """
    theta: 1-d array with shape (2,)
    X: 2-d array with shape (12, 2)
    y: 2-d array with shape (12, 1)
    l: lambda constant
    grad has same shape as theta (2,)
    """
    grad = (X @ theta - y.flatten()) @ X
    regterm = l * theta
    regterm[0] = 0  # #don't regulate bias term
    return (grad + regterm) / len(X)


def trainLinearReg(X, y, l):
    theta = np.zeros(X.shape[1])
    res = opt.minimize(fun=costReg,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=gradientReg)
    return res.x


def plot_learning_curve(X, y, Xval, yval, l):
    """画出学习曲线，即交叉验证误差和训练误差随样本数量的变化的变化"""
    xx = range(1, len(X) + 1)  # at least has one example
    training_cost, cv_cost = [], []
    for i in xx:
        res = trainLinearReg(X[:i], y[:i], l)
        training_cost_i = costReg(res, X[:i], y[:i], 0)
        cv_cost_i = costReg(res, Xval, yval, 0)
        training_cost.append(training_cost_i)
        cv_cost.append(cv_cost_i)

    plt.figure(figsize=(8, 5))
    plt.plot(xx, training_cost, label='training cost')
    plt.plot(xx, cv_cost, label='cv cost')
    plt.legend()
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.title('Learning curve for linear regression')
    plt.grid(True)
    plt.show()


path = 'ex5data1.mat'
data = loadmat(path)
# Training set
X, y = data['X'], data['y']
# Cross validation set
Xval, yval = data['Xval'], data['yval']
# Test set
Xtest, ytest = data['Xtest'], data['ytest']
# Insert a column of 1's to all of the X's, as usual
X = np.insert(X, 0, 1, axis=1)
Xval = np.insert(Xval, 0, 1, axis=1)
Xtest = np.insert(Xtest, 0, 1, axis=1)
# print('X={},y={}'.format(X.shape, y.shape))
# print('Xval={},yval={}'.format(Xval.shape, yval.shape))
# print('Xtest={},ytest={}'.format(Xtest.shape, ytest.shape))

# plotData(X,y)

theta = np.ones(X.shape[1])
# print(costReg(theta, X, y, 1))  # 303.9931922202643
# print(gradientReg(theta,X,y,1))


# fit_theta = trainLinearReg(X,y,1)

# print(costReg(fit_theta,X,y))
# plotData(X,y)
# plt.plot(X[:,1], X @ fit_theta)
#
# plt.show()
plot_learning_curve(X,y,Xval,yval,l=1)