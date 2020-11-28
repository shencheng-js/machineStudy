# -*- coding: utf-8 -*-
# @Time : 2020-11-28 20:53 
# @Author : shen
# @File : m_one.py 
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize

def load_data(path):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    return X,y

def plot_an_image(X):
    """
    随机打印一个数字
    """
    pick_one = np.random.randint(0, 5000)
    image = X[pick_one, :]  #取某一行的image
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)), cmap='gray_r')
    plt.xticks([])  # 去除刻度，美观
    plt.yticks([])
    plt.show()
    print('this should be {}'.format(y[pick_one]))


def plot_100_image(X):
    """
    随机画100个数字
    """
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)  # 随机选100个样本
    sample_images = X[sample_idx, :]  # (100,400)

    #注意画图的参数
    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))

    for row in range(10):
        for column in range(10):
            ax_array[row, column].matshow(sample_images[10 * row + column].reshape((20, 20)),
                                          cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def regularized_cost(theta, X, y, l):
    """
    don't penalize theta_0
    args:
        X: feature matrix, (m, n+1) # 插入了x0=1
        y: target vector, (m, )
        l: lambda constant for regularization
    """

    #用thetaReg去掉0参数，
    thetaReg = theta[1:]
    first = (-y*np.log(sigmoid(X@theta))) + (y-1)*np.log(1-sigmoid(X@theta))
    reg = (thetaReg@thetaReg)*l / (2*len(X))
    return np.mean(first) + reg


def regularized_gradient(theta, X, y, l):
    """
    don't penalize theta_0
    args:
        l: lambda constant
    return:
        a vector of gradient
    """
    thetaReg = theta[1:]
    first = (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)
    # 这里人为插入一维0，使得对theta_0不惩罚，方便计算
    reg = np.concatenate([np.array([0]), (l / len(X)) * thetaReg])
    #对于theta_0加上的就是0，也就是没有惩罚
    return first + reg


def one_vs_all(X, y, l, K):
    """generalized logistic regression
    args:
        X: feature matrix, (m, n+1) # with incercept x0=1
        y: target vector, (m, )
        l: lambda constant for regularization
        K: numbel of labels
    return: trained parameters
    """
    all_theta = np.zeros((K, X.shape[1]))  # (10, 401)

    for i in range(1, K + 1):
        theta = np.zeros(X.shape[1])
        y_i = np.array([1 if label == i else 0 for label in y])

        ret = minimize(fun=regularized_cost, x0=theta, args=(X, y_i, l), method='TNC',
                       jac=regularized_gradient, options={'disp': True})
        all_theta[i - 1, :] = ret.x

    return all_theta


def predict_all(X, all_theta):
    # compute the class probability for each class on each training instance
    h = sigmoid(X @ all_theta.T)  # 注意的这里的all_theta需要转置
    # create array of the index with the maximum probability
    # Returns the indices of the maximum values along an axis.
    h_argmax = np.argmax(h, axis=1)
    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1

    return h_argmax


raw_X, raw_y = load_data('ex3data1.mat')
X = np.insert(raw_X, 0, 1, axis=1) # (5000, 401)
y = raw_y.flatten()  # 这里消除了一个维度，方便后面的计算 or .reshape(-1) （5000，）

all_theta = one_vs_all(X, y, 1, 10)
# print(all_theta)  # 每一行是一个分类器的一组参数

y_pred = predict_all(X, all_theta)
accuracy = np.mean(y_pred == y)
print ('accuracy = {0}%'.format(accuracy * 100))