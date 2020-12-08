# -*- coding: utf-8 -*-
# @Time : 2020-12-3 20:19 
# @Author : shen
# @File : o_one.py 
# @Software: PyCharm


#
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


def plotData():
    """瞧一瞧数据长啥样"""
    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 1:], y, c='r', marker='x')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.grid(True)


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

def genPolyFeatures(X, power):#获取高次方的值
    """添加多项式特征
    每次在array的最后一列插入第二列的i+2次方（第一列为偏置）
    从二次方开始开始插入（因为本身含有一列一次方）
    """
    Xpoly = X.copy()
    for i in range(2, power + 1):#高次方拟合
        Xpoly = np.insert(Xpoly, Xpoly.shape[1], np.power(Xpoly[:,1], i), axis=1)
    return Xpoly

def get_means_std(X):
    """获取训练集的均值和误差，用来标准化所有数据。"""
    means = np.mean(X,axis=0)
    stds = np.std(X,axis=0,ddof=1)  # ddof=1 means 样本标准差
    return means, stds

def featureNormalize(myX, means, stds):
    """标准化"""
    X_norm = myX.copy()
    X_norm[:,1:] = X_norm[:,1:] - means[1:]
    X_norm[:,1:] = X_norm[:,1:] / stds[1:]
    return X_norm

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

def plot_lambda():
    lambdas = [0., 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1., 3., 10.]
    errors_train, errors_val = [], []
    for l in lambdas:
        theta = trainLinearReg(X_norm, y, l)
        errors_train.append(costReg(theta, X_norm, y, 0))  # 记得把lambda = 0
        errors_val.append(costReg(theta, Xval_norm, yval, 0))

    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, errors_train, label='Train')
    plt.plot(lambdas, errors_val, label='Cross Validation')
    plt.legend()
    plt.xlabel('lambda')
    plt.ylabel('Error')
    plt.grid(True)
    plt.show()

def plot_fit(means, stds, l):
    """画出拟合曲线"""
    theta = trainLinearReg(X_norm, y, l)
    x = np.linspace(-75, 55, 50)
    xmat = x.reshape(-1, 1)
    xmat = np.insert(xmat, 0, 1, axis=1)
    Xmat = genPolyFeatures(xmat, power)
    Xmat_norm = featureNormalize(Xmat, means, stds)

    plotData()
    plt.plot(x, Xmat_norm @ theta, 'b--')

path = 'ex5data1.mat'
data = loadmat(path)
# 训练集
X, y = data['X'], data['y']
# 交叉验证集
Xval, yval = data['Xval'], data['yval']
# 测试集
Xtest, ytest = data['Xtest'], data['ytest']
# 在数据中插入为1的第0列
X = np.insert(X, 0, 1, axis=1)
Xval = np.insert(Xval, 0, 1, axis=1)
Xtest = np.insert(Xtest, 0, 1, axis=1)

#查看数据属性
# print('X={},y={}'.format(X.shape, y.shape))
# print('Xval={},yval={}'.format(Xval.shape, yval.shape))
# print('Xtest={},ytest={}'.format(Xtest.shape, ytest.shape))


# #这里是拟合一次函数
# fit_theta = trainLinearReg(X,y,1)
#
# #画出拟合后的一次函数
# plotData(X,y)
# plt.plot(X[:,1], X @ fit_theta)
# plt.show()

power = 6  # 扩展到x的6次方
#对数据进行处理
train_means, train_stds = get_means_std(genPolyFeatures(X,power))
#对数据集进行规整化
X_norm = featureNormalize(genPolyFeatures(X,power), train_means, train_stds)
Xval_norm = featureNormalize(genPolyFeatures(Xval,power), train_means, train_stds)
Xtest_norm = featureNormalize(genPolyFeatures(Xtest,power), train_means, train_stds)

#训练并且画图
l=2
plot_fit(train_means, train_stds, l)
plt.show()
plot_learning_curve(X_norm, y, Xval_norm, yval, l)

plot_lambda()
#画不同样本数量的学习率
# plot_learning_curve(X,y,Xval,yval,l=1)

