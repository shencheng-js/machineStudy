# -*- coding: utf-8 -*-
# @Time : 2020-11-29 17:25 
# @Author : shen
# @File : n_one.py 
# @Software: PyCharm

#这是一个有400输入层，25的隐层，10的输出层，故而有401*25，26*10两个矩阵

#在上面那个多分类的里面有了利用训练好的模型进行前向传播，这里则是训练的过程

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.metrics import classification_report  # 这个包是评价报告
from sklearn.preprocessing import OneHotEncoder  #这个是将十进制转化为onehot形式如6化为[0,0,0,0,0,0,1,0,0,0,0]

def load_mat(path):
    '''读取数据'''
    data = loadmat(path)  # return a dict
    X = data['X']
    y = data['y'].flatten()

    return X, y

def plot_100_images(X):
    """随机画100个数字"""
    index = np.random.choice(range(5000), 100)
    images = X[index]
    fig, ax_array = plt.subplots(10, 10, sharey=True, sharex=True, figsize=(8, 8))
    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(images[r*10 + c].reshape(20,20), cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def expand_y(y):
    result = []
    # 把y中每个类别转化为一个向量，对应的lable值在向量对应位置上置为1
    for i in y:
        y_array = np.zeros(10)
        y_array[i-1] = 1   #在对应位置放1
        result.append(y_array)
    '''
    # 或者用sklearn中OneHotEncoder函数
    encoder =  OneHotEncoder(sparse=False)  # return a array instead of matrix
    y_onehot = encoder.fit_transform(y.reshape(-1,1))
    return y_onehot
    '''
    return np.array(result)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#激活函数的梯度下降，激活函数的特殊性质，反向传播算法
def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

#随机初始化，获取小范围内的随机数组
def random_init(size):
    '''从服从的均匀分布的范围中随机返回size大小的值'''
    return np.random.uniform(-0.12, 0.12, size)


def feed_forward(theta, X, ):  #进行一次正向传播，过程中得到的值
    '''得到每层的输入和输出'''
    t1, t2 = deserialize(theta)
    # 前面已经插入过偏置单元，这里就不用插入了
    a1 = X
    z2 = a1 @ t1.T
    a2 = np.insert(sigmoid(z2), 0, 1, axis=1)
    z3 = a2 @ t2.T
    a3 = sigmoid(z3)#最终输出的结果

    return a1, z2, a2, z3, a3

def cost(theta, X, y):
    a1, z2, a2, z3, h = feed_forward(theta, X)
    J = 0
    for i in range(len(X)):#累加5000个样本的误差
        first = - y[i] * np.log(h[i])
        second = (1 - y[i]) * np.log(1 - h[i])
        J = J + np.sum(first - second)
    J = J / len(X)#最后将十个可能性的代价统一返回
    return J
'''
     # or just use verctorization
     J = - y * np.log(h) - (1 - y) * np.log(1 - h)
     return J.sum() / len(X)
'''

def regularized_cost(theta, X, y, l=1):
    '''正则化时忽略每层的偏置项，也就是参数矩阵的第一列'''
    t1, t2 = deserialize(theta)
    #正则化误差项，对每层的矩阵除去偏置项以外，进行惩罚，sum是对所提供的矩阵的每一个元素求和
    reg = np.sum(t1[:,1:] ** 2) + np.sum(t2[:,1:] ** 2)  # or use np.power(a, 2)
    return l / (2 * len(X)) * reg + cost(theta, X, y)


def gradient(theta, X, y):
    '''
    unregularized gradient, notice no d1 since the input layer has no error
    return 所有参数theta的梯度，故梯度D(i)和参数theta(i)同shape，重要。
    '''
    t1, t2 = deserialize(theta)
    a1, z2, a2, z3, h = feed_forward(theta, X)
    d3 = h - y  # (5000, 10)   #最后一层的误差
    d2 = d3 @ t2[:, 1:] * sigmoid_gradient(z2)  # (5000, 25)
    D2 = d3.T @ a2  # (10, 26)
    D1 = d2.T @ a1  # (25, 401)
    D = (1 / len(X)) * serialize(D1, D2)  # (10285,)

    return D


# 正则化神经网络
def regularized_gradient(theta, X, y, l=1):
    """不惩罚偏置单元的参数"""
    D2, D3 = deserialize(gradient(theta, X, y))
    t1, t2 = deserialize(theta)
    t1[:, 0] = 0  # 即把偏置单元的参数变成0，避免后续加数时对偏置进行改变
    t2[:, 0] = 0
    reg_D2 = D2 + (l / len(X)) * t1
    reg_D3 = D3 + (l / len(X)) * t2
    return serialize(reg_D2, reg_D3)

def nn_training(X, y):
    init_theta = random_init(10285)  # 25*401 + 10*26

    res = opt.minimize(fun=regularized_cost,
                       x0=init_theta,
                       args=(X, y, 1),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'maxiter': 400})
    return res

def accuracy(theta, X, y):
    _, _, _, _, h = feed_forward(result.x, X)
    y_pred = np.argmax(h, axis=1) + 1
    print(classification_report(y, y_pred))

# 展开参数  使用高级优化方法来优化神经网络时，我们需要将多个参数矩阵展开，再恢复
def serialize(a, b):
    return np.r_[a.flatten(), b.flatten()]  # np.r指连接两个展开的array


def deserialize(seq):  # 再恢复参数。取决于几层神经网络，每层各有几个激活值,   上面的serialize先将25*401与10*26合并，到这里再拆
    return seq[:25 * 401].reshape(25, 401), seq[25 * 401:].reshape(10, 26)

def plot_hidden(theta):
    t1, _ = deserialize(theta)
    t1 = t1[:, 1:]
    fig,ax_array = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(6,6))
    for r in range(5):
        for c in range(5):
            ax_array[r, c].matshow(t1[r * 5 + c].reshape(20, 20), cmap='gray_r')
            plt.xticks([])
            plt.yticks([])
    plt.show()
#处理了数据
raw_X, raw_y = load_mat('ex4data1.mat')  #raw代表原始，最初版本
X = np.insert(raw_X, 0, 1, axis=1)     #在第0列插入1
y = expand_y(raw_y)     #获得了向量

print(X.shape,y.shape)
result = nn_training(X,y)
accuracy(result.x, X, raw_y)
plot_hidden(result.x)
# a = np.matrix([[1,2,3],[4,5,6]])
# b = np.matrix([[6,7,10],[8,9,0]])
# print(serialize(a,b))