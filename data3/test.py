#!C:\pythonCode
# -*- coding: utf-8 -*-
# @Time : 2022/10/15 18:37
# @Author : zmz
# @File : test.py
# @Software: PyCharm
import numpy as np
from scipy import io

mat = io.loadmat('point1.mat')
point1 = dict([(key, mat[key]) for key in ['wrd', 'zmz', 'zy', 'realgps']])
sortData = np.sort(point1['wrd'].flatten())
# print(sortData)
rss1 = sortData[sortData.size - 8: sortData.size - 2]  # dBm
# print(rss1)
# print(sortData)
# print(sortData.size)
# print(rss1-rss1)

temp1 = np.full(10, 100) - sortData
# print(temp1)
# print(np.mean(temp1))
less1 = np.array([1, 2, 3, 4, 5, 6])
# print(less1, type(less1))
# print(np.power(10, less1 / 1))

receivegps = io.loadmat('receivegps.mat')
# print(len(receivegps['gpszmz']))
# print(receivegps['gpszmz'][:, 0])
# print(receivegps['gpszmz'][:, 1])
# print(receivegps['gpszmz'][:, 1][0])
# print(type(np.array((1, 2))))
# print(np.array([[1,2],[1000,0]]))
# print(-np.abs(0.4 - 100))
#
# print(np.linalg.norm(np.array([2, 2]) - np.array([[3, 3],[0,0]])[0,:]))
# a = np.array([[3, 3], [0, 0]])
# print(a)
# a[0, :] = np.array([100, 10])
# print(a)
# print(np.square(2))


