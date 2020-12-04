# -*- coding: utf-8 -*-
# @Time : 2020-11-29 12:07 
# @Author : shen
# @File : logical.py 
# @Software: PyCharm

import pandas as pd
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def mp(X, theta):
    result = sigmoid(X.T @ theta)
    return result

def compute(k):
    if k > 0.9:
        return True
    elif k < 0.1:
        return False


if __name__ == "__main__":
    theta = np.array([-20, 30, 30, -40, 25, 25, 10, -20, 0])
    theta = theta.reshape([3, 3])

    print("请选择逻辑运算类型：")
    type = input("1.或运算；2，且运算；3，非运算，4，待定")
    type = int(type)
    print("请输入逻辑符号1/0")
    flag1 = input("请输入第一个符号并按下回车确认")
    flag1 = int(flag1)
    if type != 3:
        flag2 = input("请输入第二个符号并按下回车确认")
        flag2 = int(flag2)
        X = np.array([1, flag1, flag2])
    else:
        X = np.array([1, flag1, 0])

    print(compute(mp(X, theta[type - 1])))
