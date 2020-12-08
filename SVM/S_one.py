# -*- coding: utf-8 -*-
# @Time : 2020-12-8 10:25 
# @Author : shen
# @File : S_one.py 
# @Software: PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm

def load_mat(path):
    '''读取数据'''
    data = loadmat(path)  # return a dict
    X = data['X']
    y = data['y'].flatten()

    return X, y
def plot_data(X,y):
    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='rainbow')
    plt.xlabel('X1')
    plt.ylabel('X2')
    # plt.legend()

def plotBoundary(clf, X):
    '''plot decision bondary'''
    x_min, x_max = X[:,0].min()*1.2, X[:,0].max()*1.1
    y_min, y_max = X[:,1].min()*1.1,X[:,1].max()*1.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z)

def gaussKernel(x1, x2, sigma):
    return np.exp(-((x1-x2)**2).sum()/(2*sigma**2))

mat = loadmat("ex6data2.mat")
X2 = mat["X"]
y2 = mat["y"]
#
# models = [svm.SVC(C, kernel='linear') for C in [1, 100]]
# clfs = [model.fit(X, y.ravel()) for model in models]
#
# title = ['SVM Decision Boundary with C = {} (Example Dataset 1'.format(C) for C in [1, 100]]
# for model,title in zip(clfs,title):
#     plt.figure(figsize=(8,5))
#     plot_data(X, y)
#     plotBoundary(model, X)
#     plt.title(title)
#
#


sigma = 0.1
gamma = np.power(sigma,-2.)/2
clf = svm.SVC(C=1, kernel='rbf', gamma=gamma)
modle = clf.fit(X2, y2.flatten())
plot_data(X2, y2)
plotBoundary(modle, X2)
plt.show()