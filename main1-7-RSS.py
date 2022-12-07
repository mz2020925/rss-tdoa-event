#!C:\pythonCode
# -*- coding: utf-8 -*-
# @Time : 2022/10/15 14:17
# @Author : zmz
# @File : main1-7-RSS.py
# @Software: PyCharm
"""
 本算法是初步选择的利用场强衰减计算距离的算法。算法思路是使用自适应三边定位，取三圆交点(3个)，以3个交点取加权质心。
 算法关键在于自适应—就是想办法让不相交的三圆相交出需要的三个点。
 接收机：wrd—1，zmz—2，zy—3

 ***目前此算法只能定位三角形圆内的点
"""
import numpy as np
from scipy import io
from otherFunctions import *
import matplotlib.pyplot as plt
from threading import Thread


def readmat(filename='point1.mat'):
    """
    :param filename:
    :return point: point是一个字典，包含这个信号源点到三个接收机接收dBm，和它本身的实际经纬度
    """
    mat = io.loadmat(filename)
    if 'realgps' in mat.keys():
        point = dict([(key, mat[key]) for key in ['wrd', 'zmz', 'zy', 'realgps']])
        # print(point)
        return point
    else:
        receivegps = dict([(key, mat[key]) for key in ['gpswrd', 'gpszmz', 'gpszy']])
        return receivegps


def main():
    # 加载数据, 信号源实际经纬度
    # point1.mat, point2.mat, point3.mat, point4.mat, point5.mat, point6.mat,point7.mat
    # point = readmat('data3/point4.mat')
    # receivegps = readmat('data3/receivegps.mat')
    
    # """三边测量法+质心法"""
    # estixy, estilonlat, realxy, reallonlat = estimatePosition(point, receivegps)
    # print(f'估算信号源位置：{estilonlat}, xy：{estixy}')
    # print(f'实际信号源位置：{reallonlat}, xy：{realxy}')
    # errpr = np.linalg.norm(estixy - realxy)
    # print(f'误差：{errpr}')


    """SOCP-K法"""
    point = readmat('data3/point4.mat')
    receivegps = readmat('data3/receivegps.mat')
    SOCP_K(point, receivegps)


if __name__ == '__main__':
    main()
