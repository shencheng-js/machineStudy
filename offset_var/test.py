# -*- coding: utf-8 -*-
# @Time : 2020-12-4 22:55 
# @Author : shen
# @File : test.py 
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt

def bafi(year):
    return np.power(1.15,year)


if __name__ == "__main__":
    cost = []
    year = np.arange(2000,2020,1)
    print(year)
    for i in range(20):
        print(str(i+2000)+"年可以获得收益："+str(bafi(i)))
        cost.append(bafi(i))

    fig = plt.figure(num=1, figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.plot(year,cost)
    plt.show()