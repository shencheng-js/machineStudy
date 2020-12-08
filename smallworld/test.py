# -*- coding: utf-8 -*-
# @Time : 2020-12-7 16:09 
# @Author : shen
# @File : test.py 
# @Software: PyCharm

def add(mat,i):
    mat[i]=10

if __name__ == "__main__":
    mat = [1,2,3]
    add(mat,0)
    print(mat)