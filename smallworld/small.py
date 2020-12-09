# -*- coding: utf-8 -*-
# @Time : 2020-12-7 15:52 
# @Author : shen
# @File : small.py 
# @Software: PyCharm

"""判断远近距离导致的影响使用使用高斯函数，就在旁边接近1，很远则为0；每次更新激活表，每个神经元累计收到l值以上方可激活"""
"""默认一轮之后无残留影响，想要实现残留影响，每轮减半之类的"""
"""最终都是全部消散或者一直被激活，所以附加偶然情况是最好的"""
import numpy as np
import matplotlib.pyplot as plt


def gaosi(x, x_i, k=5.5):  # k值是σ值，40对范围10以内的神经元有影响
    return np.exp(-1.0 * np.power((x - x_i), 2) / k * 2)


def activ(yuan, i):  # 对其内i附近神经元进行影响,注意自己对自己不要影响
    for j in range(0, 100):
        if i == j:
            continue
        result = gaosi(i, j)
        # print(result)
        yuan[i][j] = yuan[i][j] + result

    return yuan


def pltgaosi():
    X = np.arange(-20, 20, 1)
    y = gaosi(0, X, 20)
    fig = plt.figure(num=1, figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.plot(X, y)
    plt.show()


def toone(temp, threshold):
    for i in range(0, 100):
        if (temp[i] >= threshold):
            temp[i] = 1
        else:
            temp[i] = 0
    return temp


def getjihuo(yuan, threshold=35, fre=0):  # 这里需要控制量值，也就是多大刺激能够激活,以及其之前的影响，默认为没有
    temp = activlis * fre
    for i in range(0, 100):
        for j in range(0, 100):
            temp[j] += yuan[i][j]

    # print(temp)
    return toone(temp, threshold)


def rand(activ, num=3):  # 随机激活神经元，控制在num个以内，可能会有重叠
    temp = np.random.randint(0, 100, size=num)
    for i in temp:
        activ[i] = 1

    return activ


def getzero():  # 获取100*100的二维数组
    ret = [([0]*100) for j in range(100)]
    # return [[0] * 100] * 100
    return ret


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
    neuron_size = 100
    activation_list = np.random.randint(low=0, high=2, size=size)

    #获取激活累加数组
    accumulation = get_zero()

    result = []
    for i in range(101):  # 训练100次
        for j in range(0, 100):
            if activlis[j] == 1:  # 如果是被激活的，刺激其他神经元
                accumulation = activ(yuan, j)  # 一次激活完成

        # print(yuan)
        # 一轮训练激活结束，找成功激活的神经元
        activation_list = getjihuo(accumulation)  # 默认之前的无影响
        # 随机激活几个神经元
        activation_list = rand(activation_list)

        print("第" + str(i) + "轮训练结束，激活的神经元有")
        print(activation_list)
        result.append(activation_list)
        accumulation = getzero()

    plot_result(data=result)
