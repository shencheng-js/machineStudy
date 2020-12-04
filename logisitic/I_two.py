# -*- coding: utf-8 -*-
# @Time : 2020-11-25 17:37 
# @Author : shen
# @File : I_two.py 
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# 对于非线性可分的数据进行处理
path = r'D:\fireFox_download\机器学习题目代码及数据文件\logistics回归\数据文件\ex2data2.txt'
data = pd.read_csv(path, header=None, names=['test1', 'test2', 'accept'])




# 一个能够将x1和x2以n次乘积
def feature_mapping(x1, x2, power):
    data = {}
    for i in np.arange(power + 1):
        for p in np.arange(i + 1):
            # 下面的data["f{}{}".format(i - p, p)]的意思是该列的colmns就是下面这个
            data["f{}{}".format(i - p, p)] = np.power(x1, i - p) * np.power(x2, p)

    return pd.DataFrame(data)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# print(feature_mapping(t1,t2,6))
def cost(X, y, theta):
    left = (-y) * np.log(sigmoid(X @ theta))
    right = (1 - y) * np.log(1 - sigmoid(X @ theta))
    return np.mean(left - right)


def costReg(theta,X, y,l=1):
    _theta = theta[1:]
    reg = (l / (2 * len(X))) * (_theta @ _theta)

    return cost(X, y, theta) + reg


def gradient(theta, X, y):
    theta = (X.T @ (sigmoid(X @ theta) - y)) / len(X)
    return theta
    # return (X.T @ (sigmoid(X @ theta) - y))/len(X)


def gradientReg(theta, X, y, l=1):
    reg = (l / len(X)) * theta
    reg[0] = 0

    return gradient(theta, X, y) + reg


t1 = data.iloc[:, 0].values
t2 = data.iloc[:, 1].values
y = data.iloc[:, -1].values
X = feature_mapping(t1, t2, 6)
theta = np.zeros(X.shape[1])

result = opt.fmin_tnc(func=costReg, x0=theta, fprime=gradientReg, args=(X, y))
print(result)

final_theta = result[0]
x = np.linspace(-1, 1.5, 250)
xx, yy = np.meshgrid(x, x)

z = feature_mapping(xx.ravel(), yy.ravel(), 6).values
z = z @ final_theta
z = z.reshape(xx.shape)

# plt.contour(xx, yy, z, 0)
# plt.ylim(-.8, 1.2)

# print(data)
positive = data[data.accept.isin(['1'])]  # 1
negetive = data[data.accept.isin(['0'])]  # 0


#作图
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(positive['test1'], positive['test2'], c='b', label='Admitted')
ax.scatter(negetive['test1'], negetive['test2'], s=50, c='r', marker='x', label='Not Admitted')
# 设置图例显示在图的上方
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
ax.legend(loc='center left', bbox_to_anchor=(0.2, 1.12), ncol=3)
# 设置横纵坐标名
ax.set_xlabel('test 1 Score')
ax.set_ylabel('test 2 Score')
plt.contour(xx, yy, z, 0)
plt.ylim(-.8, 1.2)
plt.show()