# -*- coding: utf-8 -*-
# @Time : 2020-11-25 19:38 
# @Author : shen
# @File : test_n.py 
# @Software: PyCharm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

if __name__ == "__main__":
    # 激活函数
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))


    x1 = np.arange(-10, 10, 0.1)
    plt.plot(x1, sigmoid(x1), c='r')
    plt.show()