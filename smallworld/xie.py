# -*- coding: utf-8 -*-
# @Time : 2020-12-9 19:07 
# @Author : shen
# @File : xie.py 
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt

def ReLU(x,x_i,power=2):#激活函数
    ret = max(0, 1 / abs(x - x_i) ** 2)
    return ret


def activ(yuan, i):  # 对其内i附近神经元进行影响,注意自己对自己不要影响
    for j in range(0, 100):
        if i == j:
            continue
        result = ReLU(i, j)
        yuan[i][j] = yuan[i][j] + result
    return yuan

def to_one(temp, threshold):#满足门槛即可激活
    for i in range(0, 100):
        if (temp[i] > threshold):
            temp[i] = 1
        else:
            temp[i] = 0
    return temp


# 用高斯就用2.3的门槛，用线性就用1.7左右
def getjihuo(yuan, activlis, threshold=1.7):  # 这里需要控制量值，也就是多大刺激能够激活,以及其之前的影响，默认为没有
    temp = [0] * 100
    for i in range(0, 100):
        for j in range(0, 100):
            # print("当前激活值"+str(yuan[i][j]))
            # print("激活前"+str(temp[j]))
            temp[j] = temp[j] + yuan[i][j]
            # print("激活后" + str(temp[j]))

    # print(temp)
    return to_one(temp, threshold)


def rand(activ, num=3):  # 随机激活神经元，控制在num个以内，可能会有重叠
    temp = np.random.randint(0, 100, size=num)
    for i in temp:
        activ[i] = 1

    return activ


def getzero():  # 获取100*100的二维数组
    x = 100
    y = 100
    retlist = []
    for i in range(x):
        retlist.append([])
        for j in range(y):
            retlist[i].append(0)
    return retlist


def plot_result(data):
    fig = plt.figure(num=1, figsize=(10, 10))
    ax = fig.add_subplot(111)
    for i in range(100):  # 第几轮循环，应该是y坐标
        for j in range(len(data[i])):
            if (data[i][j] == 1):  # 如果被激活，就画点
                ax.scatter(j, i)

    ax.set_xlabel("Activated neurons", fontsize=16)
    ax.set_ylabel("Frequency of training", fontsize=16)
    plt.show()


if __name__ == "__main__":
    # pltgaosi()
    size = 100
    activlis = np.random.randint(low=0, high=2, size=size)
    yuan = getzero()

    result = []#统计结果，即每一轮训练情况
    result.append(activlis)

    for i in range(100):  # 训练100次
        for j in range(0, 100):
            if activlis[j] == 1:  # 如果是被激活的，刺激其他神经元
                yuan = activ(yuan, j)  # 一次激活完成

        # print(yuan)
        # 一轮训练激活结束，找成功激活的神经元
        activlis = getjihuo(yuan=yuan, activlis=activlis)  # 默认之前的无影响
        # 随机激活几个神经元
        # activlis = rand(activlis)

        print("第" + str(i) + "轮训练结束，神经元情况为有")
        print(activlis)

        result.append(activlis)
        yuan = getzero()

    plot_result(data=result)
