# -*- coding: utf-8 -*-
# @Time : 2020-12-7 16:09 
# @Author : shen
# @File : test.py 
# @Software: PyCharm

import pandas as pd
import numpy as np

def add():
    x = 100
    y = 100
    retlist = []
    for i in range(x):
        retlist.append([])
        for j in range(y):
            retlist[i].append(0)
    return retlist

if __name__ == "__main__":
    data = add()
    print(len(data[0]))