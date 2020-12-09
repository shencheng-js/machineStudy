# -*- coding: utf-8 -*-
# @Time : 2020-12-9 20:11 
# @Author : shen
# @File : final_2018213264.py 
# @Software: PyCharm

# -*- coding: utf-8 -*-
# @Time : 2020-12-9 18:13
# @Author : shen
# @File : small_my.py
# @Software: PyCharm

"""判断远近距离导致的影响使用使用高斯函数，就在旁边接近1，很远则为0；每次更新激活表，每个神经元累计收到l值以上方可激活"""
"""默认一轮之后无残留影响，想要实现残留影响，每轮减半之类的"""
"""最终都是全部消散或者一直被激活，所以附加偶然情况是最好的"""
import numpy as np
import matplotlib.pyplot as plt

def Gaussian_activation(x, x_i, k=0.4):  # k值是σ值，0.5,越大影响范围越小
    ret = np.exp(-1.0 * np.power((x - x_i), 2) / 2 * k ** 2)
    # ret = max(0, 1 / abs(x - x_i) ** 2)
    return ret


def discharging(temp_accumulation, i):  # 对其内i附近神经元进行影响,注意自己对自己不要影响
    for j in range(0, 100):
        if i == j:
            continue
        result = Gaussian_activation(i, j)
        temp_accumulation[i][j] = temp_accumulation[i][j] + result
    return temp_accumulation


def pltGaussian_activation():
    X = np.arange(-20, 20, 1)
    y = Gaussian_activation(0, X)
    fig = plt.figure(num=1, figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.plot(X, y)
    plt.show()


def to_one(temp, threshold):
    for i in range(0, 100):

        if (temp[i] > threshold):
            temp[i] = 1
        else:
            temp[i] = 0
    return temp


# 用高斯就用1.65的门槛
def Concentrated_discharge(temp_accumulation, activlis, threshold=2.64, fre=1):  # 这里需要控制量值，也就是多大刺激能够激活,以及其之前的影响，默认为没有
    temp = [0] * 100
    for i in range(0, 100):
        for j in range(0, 100):
            temp[j] = temp[j] + temp_accumulation[i][j]

    return to_one(temp, threshold)


def rand_add(list_temp, num=3):  # 随机激活神经元，控制在num个以内，可能会有重叠
    temp = np.random.randint(0, 100, size=num)
    for i in temp:
        list_temp[i] = 1

    return list_temp


def get_100_zero():  # 获取100*100的二维数组
    x = 100
    y = 100
    retlist = []
    for i in range(x):
        retlist.append([])
        for j in range(y):
            retlist[i].append(0)
    return retlist


def plot_result(data, time = np.random.randint(0,1000,1)):
    fig = plt.figure(num=1, figsize=(10, 10))
    ax = fig.add_subplot(111)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    for i in range(100):  # 第几轮循环，应该是y坐标
        for j in range(len(data[i])):
            if (data[i][j] == 1):  # 如果被激活，就画点
                ax.scatter(j, i)

    ax.set_xlabel("100个神经元状态(激活则加点)", fontsize=26)
    ax.set_ylabel("训练次数", fontsize=26)
    ax.set_title("训练情况", fontsize=36)
    plt.savefig("./" + "随机测试的第" + str(time) + "次训练结果" + ".jpg", dpi=200)
    plt.show()


if __name__ == "__main__":
    neuron_size = 100
    activation_list = np.random.randint(low=0, high=2, size=neuron_size)

    #获取神经元刺激的累加值
    accumulation = get_100_zero()

    final_result = []
    final_result.append(activation_list)


    for i in range(100):  # 训练100次
        for j in range(0, 100):
            if activation_list[j] == 1:  # 如果是被激活的，刺激其他神经元
                accumulation = discharging(accumulation, j)  # 一次激活完成

        # 一轮训练激活结束，找成功激活的神经元
        activation_list = Concentrated_discharge(accumulation, activlis=activation_list)  # 默认之前的无影响
        # 随机激活几个神经元
        # activation_list = rand_add(activation_list)

        print("第" + str(i) + "轮训练结束，激活的神经元有")
        print(activation_list)
        final_result.append(activation_list)
        accumulation = get_100_zero()

    plot_result(data=final_result)