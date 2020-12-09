# -*- coding: utf-8 -*-
# @Time : 2020-12-8 19:06 
# @Author : shen
# @File : test.py 
# @Software: PyCharm

import scipy.io as sio

import numpy as np

import pandas as pd

file_path = r"E:\认知\认知科学与基础\Datasets\DEAP\data_preprocessed_matlab\s01.mat"
mat = sio.loadmat(file_path)
data = mat['data']
labels = mat['labels']

# print(data.shape,labels.shape)#(40, 40, 8064) (40, 4)
valence_labels = []
# print(labels)
for i in range(len(labels[:, 0])):
    if (labels[i, 0] <= 5):
        valence_labels.append(0)
    else:
        valence_labels.append(1)
print(valence_labels)

arousal_labels = []
for i in range(len(labels[:, 1])):
    if (labels[i, 1] <= 5):
        arousal_labels.append(0)
    else:
        arousal_labels.append(1)
print(arousal_labels)


def calc_features(data):#获取data里面的各个性质，平均值之类的
    result = []
    result.append(np.mean(data))
    result.append(np.median(data))
    result.append(np.max(data))
    result.append(np.min(data))
    result.append(np.std(data))
    result.append(np.var(data))
    result.append(np.max(data) - np.min(data))
    result.append(pd.Series(data).skew)
    result.append(pd.Series(data).kurt)

    print("+++++++++++++++++++++++")
    print(result)
    print("=========================")
    # result = np.array(result)
    #
    # print(type(result))
    return result


data = data[:, :, 128 * 3:]

featured_data = np.zeros([40, 40, 101])
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        for k in range(10):
            print(data[i, j,(k * 128 * 6):((k + 1) * 128 * 6)])
            featured_data[i, j, k * 9:((k + 1) * 9)]= calc_features(data[i, j,(k * 128 * 6):((k + 1) * 128 * 6)])
            featured_data[i, j, 10 * 9:11 * 9] = calc_features(data[i, j, :])
            featured_data[i, j, 99] = j
            featured_data[i, j, 100] = 1

print(featured_data.shape)
print(valence_labels.shape)
print(arousal_labels.shape)
