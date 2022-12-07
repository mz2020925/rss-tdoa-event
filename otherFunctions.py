#!C:\pythonCode
# -*- coding: utf-8 -*-
# @Time : 2022/10/15 20:34
# @Author : zmz
# @File : otherFunctions.py
# @Software: PyCharm
from cmath import log10
import math
from tkinter import E
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


# 此函数实现了SOCPK算法
def SOCP_K(point, receivegps, send=100):
    # 计算接收机dBm，并计算损耗，然后计算距离
    # 数据处理并计算接收机dBm，从10组数据中提取6组数据取平均。也可以取最大值（另一种方式）
    sortData = np.sort(point['wrd'].flatten())
    rss1 = sortData[2: len(sortData) - 2]  # dBm
    sortData = np.sort(point['zmz'].flatten())
    rss2 = sortData[2: len(sortData) - 2]  # dBm
    sortData = np.sort(point['zy'].flatten())
    rss3 = sortData[2: len(sortData) - 2]  # dBm

    # 功率差来计算损耗
    number = len(rss1)
    less1 = np.full(number, send) - (rss1 + np.full(number, 107))
    less2 = np.full(number, send) - (rss2 + np.full(number, 107))
    less3 = np.full(number, send) - (rss3 + np.full(number, 107))

    # 根据拟合出的n，估算出信号源到接收机的距离
    n1 = -0.02972 * np.mean(rss1) + 0.854
    r1arr = np.power(10, less1 / (10 * n1))
    r1 = np.mean(r1arr)

    n2 = -0.02972 * np.mean(rss2) + 0.854
    r2arr = np.power(10, less2 / (10 * n2))
    r2 = np.mean(r2arr)

    n3 = -0.02972 * np.mean(rss3) + 0.854
    r3arr = np.power(10, less3 / (10 * n3))
    r3 = np.mean(r3arr)
    print(f'原始半径:r1={r1},r2={r2},r3={r3}')

    # 计算接收机经纬度
    sortgps = np.sort(receivegps['gpswrd'][:, 0])
    long1 = np.mean(sortgps[2:len(sortgps) - 2])
    sortgps = np.sort(receivegps['gpswrd'][:, 1])
    lati1 = np.mean(sortgps[2:len(sortgps) - 2])

    sortgps = np.sort(receivegps['gpszmz'][:, 0])
    long2 = np.mean(sortgps[2:len(sortgps) - 2])
    sortgps = np.sort(receivegps['gpszmz'][:, 1])
    lati2 = np.mean(sortgps[2:len(sortgps) - 2])

    sortgps = np.sort(receivegps['gpszy'][:, 0])
    long3 = np.mean(sortgps[2:len(sortgps) - 2])
    sortgps = np.sort(receivegps['gpszy'][:, 1])
    lati3 = np.mean(sortgps[2:len(sortgps) - 2])

    # 计算接收机相对坐标，以long3,lati3（zy位置）作为参考原点(0,0)
    x1, y1 = lonlat2xy(long1, lati1, long3, lati3)
    x2, y2 = lonlat2xy(long2, lati2, long3, lati3)
    x3, y3 = lonlat2xy(long3, lati3, long3, lati3)

    # 计算接收机构成的三角形的边长
    d12 = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
    d23 = np.linalg.norm(np.array([x2, y2]) - np.array([x3, y3]))
    d13 = np.linalg.norm(np.array([x1, y1]) - np.array([x3, y3]))

    # 信号源实际经纬度
    reallon = np.mean(point['realgps'][:, 0])
    reallat = np.mean(point['realgps'][:, 1])
    reallonlat = np.array([reallon, reallat])

    # 信号源实际xy坐标
    realx, realy = lonlat2xy(reallonlat[0], reallonlat[1], long3, lati3)
    realxy = np.array([realx, realy])

    """开始SOCP-K算法"""
    # 初始公式变量
    L_ij_A = np.array([np.mean(less1), np.mean(less2), np.mean(less3)])  # 从发射源到三个接收机的功率损耗 -- L_11_A、L_12_A、L_13_A
    # print(L_ij_A)
    L0 = np.array(-7.0)  # 从发射源到距发射源 1m 处的功率损耗
    gamma = np.array([n1, n2, n3])  # 三个路径损耗因子
    # 三个接收机坐标
    s1= np.array([x1, y1]).T
    s2= np.array([x2, y2]).T
    s3= np.array([x3, y3]).T
    # s123 = np.array([s1, s2, s3])  # 3行2列的接收机坐标矩阵   
    # 距接收机1m
    d0 = np.array(1.0)
    sigma = np.array(3.981)  # 接收功率的误差是服从正态分布的，这是它的方差

    # 待优化的函数公式是一个非凸非线性，需要变换公式
    # 第一次变换没有引入新的变量
    # 第二次变换
    Alpha_ij_A = np.power(10, np.full(3,L0-L_ij_A)/(10*gamma))  # Alpha_ij_A -- Alpha_11_A、Alpha_12_A、Alpha_13_A
    # 第三次变换
    mu = d0*(np.log(10)/(10*gamma))  # mu由于gamma是一个行向量，也变成了一个向量

    # 开始定义算法所需变量
    # x1x = np.array(0.0)
    # x1y = np.array(0.0)
    # X = np.array([[x1x, x1y]]).T
    # y = X.reshape((-1,1),order='F')  # 实际上 y = X 即可
    # I_2M = np.array([[1,0],[0,1]])
    Ei = np.array([[1,0],[0,1]])

    f = np.array([1,1,1])

    # Define and solve the CVXPY problem.
    # 待求解的 SOCP变量，1行3列的行向量
    hij= cp.Variable(3)
    # 待求解的 估计信号源位置
    x1x_x1y = np.random.randn(1, 2).T
    # 中间变量
    d11_A = x1x_x1y.T @ x1x_x1y - 2*s1.T @ x1x_x1y + np.linalg.norm(s1,2)
    d12_A = x1x_x1y.T @ x1x_x1y - 2*s2.T @ x1x_x1y + np.linalg.norm(s2,2)
    d13_A = x1x_x1y.T @ x1x_x1y - 2*s3.T @ x1x_x1y + np.linalg.norm(s3,2)
    A = []
    A.append([2*(1/mu[0])*(d0*d0*(1/Alpha_ij_A[0]) - Alpha_ij_A[0]*d11_A), 4*d11_A*sigma*sigma - hij[0]])
    A.append([2*(1/mu[1])*(d0*d0*(1/Alpha_ij_A[1]) - Alpha_ij_A[1]*d12_A), 4*d12_A*sigma*sigma - hij[1]])
    A.append([2*(1/mu[2])*(d0*d0*(1/Alpha_ij_A[2]) - Alpha_ij_A[2]*d13_A), 4*d13_A*sigma*sigma - hij[2]])
    print(A)
    # print(d11_A)
    # print(4*d11_A*sigma + hij[0])
    # print(np.array([[2*(1/mu[0])*(d0*d0*(1/Alpha_ij_A[0]) - Alpha_ij_A[0]*d11_A)], [4*d11_A*sigma*sigma - hij[0]]]))

    
    # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
    soc_constraints = [
        d11_A == x1x_x1y.T @ x1x_x1y - 2*s1.T @ x1x_x1y + np.linalg.norm(s1,2),
        cp.SOC(hij[0] + 4*d11_A*sigma, np.array([[2*(1/mu[0])*(d0*d0*(1/Alpha_ij_A[0]) - Alpha_ij_A[0]*d11_A)], [4*d11_A*sigma*sigma - hij[0]]])),

        d12_A == x1x_x1y.T @ x1x_x1y - 2*s2.T @ x1x_x1y + np.linalg.norm(s2,2),
        cp.SOC(4*d12_A*sigma + hij[1], np.array([[2*(1/mu[1])*(d0*d0*(1/Alpha_ij_A[1]) - Alpha_ij_A[1]*d12_A)], [4*d12_A*sigma*sigma - hij[1]]])),

        d13_A == x1x_x1y.T @ x1x_x1y - 2*s3.T @ x1x_x1y + np.linalg.norm(s3,2),
        cp.SOC(4*d13_A*sigma + hij[2], np.array([[2*(1/mu[2])*(d0*d0*(1/Alpha_ij_A[2]) - Alpha_ij_A[2]*d13_A)], [4*d13_A*sigma*sigma - hij[2]]])),

        # cp.SOC(hij + np.full(3,4*d13_A*sigma), np.array([[2*(1/mu[2])*(d0*d0*(1/Alpha_ij_A[2]) - Alpha_ij_A[2]*d13_A)], [4*d13_A*sigma*sigma - hij[2]]]))
    ]
    prob = cp.Problem(cp.Minimize(f.T @ hij), soc_constraints)
    prob.solve()

    # Print result.
    print("The optimal value is", prob.value)
    print("A solution x is")
    print(hij.value)
    print(soc_constraints[1].dual_value)
    print(soc_constraints[3].dual_value)
    print(soc_constraints[5].dual_value)
    


def RSOCPU():
    pass


# 此函数实现了三边测量法 + 加权质心法
def estimatePosition(point, receivegps, send=100):
    # 计算接收机dBm，并计算损耗，然后计算距离
    # 数据处理并计算接收机dBm，从10组数据中提取6组数据取平均。也可以取最大值--另一种方式
    sortData = np.sort(point['wrd'].flatten())
    rss1 = sortData[2: len(sortData) - 2]  # dBm
    sortData = np.sort(point['zmz'].flatten())
    rss2 = sortData[2: len(sortData) - 2]  # dBm
    sortData = np.sort(point['zy'].flatten())
    rss3 = sortData[2: len(sortData) - 2]  # dBm

    # 功率差来计算损耗
    number = len(rss1)
    less1 = np.full(number, send) - (rss1 + np.full(number, 107))
    less2 = np.full(number, send) - (rss2 + np.full(number, 107))
    less3 = np.full(number, send) - (rss3 + np.full(number, 107))

    # 根据拟合出的n，估算出信号源到接收机的距离
    n1 = -0.02972 * np.mean(rss1) + 0.854
    r1arr = np.power(10, less1 / (10 * n1))
    r1 = np.mean(r1arr)

    n2 = -0.02972 * np.mean(rss2) + 0.854
    r2arr = np.power(10, less2 / (10 * n2))
    r2 = np.mean(r2arr)

    n3 = -0.02972 * np.mean(rss3) + 0.854
    r3arr = np.power(10, less3 / (10 * n3))
    r3 = np.mean(r3arr)
    print(f'原始半径:r1={r1},r2={r2},r3={r3}')

    # 计算接收机经纬度
    sortgps = np.sort(receivegps['gpswrd'][:, 0])
    long1 = np.mean(sortgps[2:len(sortgps) - 2])
    sortgps = np.sort(receivegps['gpswrd'][:, 1])
    lati1 = np.mean(sortgps[2:len(sortgps) - 2])

    sortgps = np.sort(receivegps['gpszmz'][:, 0])
    long2 = np.mean(sortgps[2:len(sortgps) - 2])
    sortgps = np.sort(receivegps['gpszmz'][:, 1])
    lati2 = np.mean(sortgps[2:len(sortgps) - 2])

    sortgps = np.sort(receivegps['gpszy'][:, 0])
    long3 = np.mean(sortgps[2:len(sortgps) - 2])
    sortgps = np.sort(receivegps['gpszy'][:, 1])
    lati3 = np.mean(sortgps[2:len(sortgps) - 2])

    # 计算接收机相对坐标，以long3,lati3（zy位置）作为参考原点(0,0)
    x1, y1 = lonlat2xy(long1, lati1, long3, lati3)
    x2, y2 = lonlat2xy(long2, lati2, long3, lati3)
    x3, y3 = lonlat2xy(long3, lati3, long3, lati3)

    # 计算接收机构成的三角形的边长
    d12 = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
    d23 = np.linalg.norm(np.array([x2, y2]) - np.array([x3, y3]))
    d13 = np.linalg.norm(np.array([x1, y1]) - np.array([x3, y3]))

    # 信号源实际经纬度
    reallon = np.mean(point['realgps'][:, 0])
    reallat = np.mean(point['realgps'][:, 1])
    reallonlat = np.array([reallon, reallat])

    # 信号源实际xy坐标
    realx, realy = lonlat2xy(reallonlat[0], reallonlat[1], long3, lati3)
    realxy = np.array([realx, realy])

    # 绘制出初始状态的三个圆
    x123 = np.array([x1, x2, x3])
    y123 = np.array([y1, y2, y3])
    r123 = np.array([r1, r2, r3])
    drawfigure(x123, y123, r123, [], realxy, np.array([0, 0]), '1.png')

    # 设置三个半径的上限
    if r1 > np.max(np.array([d12, d13])):
        r1 = np.max(np.array([d12, d13]))
    if r2 > np.max(np.array([d12, d23])):
        r2 = np.max(np.array([d12, d23]))
    if r3 > np.max(np.array([d23, d13])):
        r3 = np.max(np.array([d23, d13]))

    # 加权质心法估算信号源xy位置
    cross12 = circleCross(x1, y1, r1, x2, y2, r2, d12)
    cross23 = circleCross(x2, y2, r2, x3, y3, r3, d23)
    cross13 = circleCross(x1, y1, r1, x3, y3, r3, d13)
    three = threeCrosspoint(x1, y1, x2, y2, x3, y3, cross12, cross23, cross13)
    print(f'三个交点：{three[0], three[1], three[2]}')
    estix = (three[0, 0] / (r1 + r2) + three[1, 0] / (r2 + r3) + three[2, 0] / (r1 + r3)) / (
            1 / (r1 + r2) + 1 / (r2 + r3) + 1 / (r1 + r3))
    estiy = (three[0, 1] / (r1 + r2) + three[1, 1] / (r2 + r3) + three[2, 1] / (r1 + r3)) / (
            1 / (r1 + r2) + 1 / (r2 + r3) + 1 / (r1 + r3))
    estixy = np.array([estix, estiy])
    estilon, estilat = xy2lonlat(estix, estiy, long3, lati3)
    estilonlat = np.array([estilon, estilat])

    # 绘制出估算后的三个圆
    drawfigure(x123, y123, np.array([r1, r2, r3]), three, realxy, estixy, '2.png')

    return estixy, estilonlat, realxy, reallonlat


# 经纬度转xy
def lonlat2xy(lon, lat, ref_lon, ref_lat):
    # input GPS and Reference GPS in degrees
    # output XY in meters (m) X:North Y:East
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)

    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    ref_sin_lat = math.sin(ref_lat_rad)
    ref_cos_lat = math.cos(ref_lat_rad)

    cos_d_lon = math.cos(lon_rad - ref_lon_rad)

    arg = np.clip(ref_sin_lat * sin_lat + ref_cos_lat * cos_lat * cos_d_lon, -1.0, 1.0)
    c = math.acos(arg)

    k = 1.0
    if abs(c) > 0:
        k = (c / math.sin(c))

    x = k * (ref_cos_lat * sin_lat - ref_sin_lat * cos_lat * cos_d_lon) * 6371393.0
    y = k * cos_lat * math.sin(lon_rad - ref_lon_rad) * 6371393.0

    return x, y


# xy转经纬度
def xy2lonlat(x, y, ref_lon, ref_lat):
    x_rad = float(x) / 6371393.0
    y_rad = float(y) / 6371393.0
    c = math.sqrt(x_rad * x_rad + y_rad * y_rad)

    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)

    ref_sin_lat = math.sin(ref_lat_rad)
    ref_cos_lat = math.cos(ref_lat_rad)

    if abs(c) > 0:
        sin_c = math.sin(c)
        cos_c = math.cos(c)

        lat_rad = math.asin(cos_c * ref_sin_lat + (x_rad * sin_c * ref_cos_lat) / c)
        lon_rad = (ref_lon_rad + math.atan2(y_rad * sin_c, c * ref_cos_lat * cos_c - x_rad * ref_sin_lat * sin_c))

        lat = math.degrees(lat_rad)
        lon = math.degrees(lon_rad)

    else:
        lat = math.degrees(ref_lat)
        lon = math.degrees(ref_lon)

    return lon, lat


# 求两相交圆的交点，返回两个交点坐标
def circleCross(x0, y0, r0, x1, y1, r1, distance):
    k1 = (y0 - y1) / (x0 - x1)
    b1 = y1 - k1 * x1

    k2 = -1.0 / k1
    b2 = (np.square(r0) - np.square(r1) - np.square(x0) + np.square(x1) - np.square(y0) + np.square(y1)) / (
            2 * (y1 - y0))
    p = np.array([])
    if distance == np.abs(r1 - r0) or distance == r1 + r0:
        xx = -(b1 - b2) / (k1 - k2)
        yy = -(-b2 * k1 + b1 * k2) / (k1 - k2)
        p = np.array([[xx, yy], [xx, yy]])
    elif np.abs(r1 - r0) < distance < r1 + r0:
        xx1 = (-b2 * k2 + x1 + k2 * y1 - np.sqrt(
            -np.square(b2) + np.square(r1) + np.square(k2) * np.square(r1) - 2 * b2 * k2 * x1 - np.square(
                k2) * np.square(x1) + 2 * b2 * y1 + 2 * k2 * x1 * y1 - np.square(y1))) / (
                      1 + np.square(k2))
        yy1 = k2 * xx1 + b2

        xx2 = (-b2 * k2 + x1 + k2 * y1 + np.sqrt(
            -np.square(b2) + np.square(r1) + np.square(k2) * np.square(r1) - 2 * b2 * k2 * x1 - np.square(
                k2) * np.square(x1) + 2 * b2 * y1 + 2 * k2 * x1 * y1 - np.square(y1))) / (
                      1 + np.square(k2))
        yy2 = k2 * xx2 + b2
        p = np.array([[xx1, yy1], [xx2, yy2]])
    return p


# 求出三圆相交时所需的三个点的坐标
def threeCrosspoint(x1, y1, x2, y2, x3, y3, cross12, cross23, cross13):
    threePoints = np.array([[0., 0.], [0., 0.], [0., 0.]])
    # 若两圆有交点,则一定是两个，因为在circleCross中判断相切时是看作两个点坐标相同
    if len(cross12) == 2:
        d1 = np.linalg.norm(cross12[0, :] - np.array([x3, y3]))
        d2 = np.linalg.norm(cross12[1, :] - np.array([x3, y3]))
        if d1 < d2:
            threePoints[0, :] = cross12[0, :]
        else:
            threePoints[0, :] = cross12[1, :]

    if len(cross23) == 2:
        d1 = np.linalg.norm(cross23[0, :] - np.array([x1, y1]))
        d2 = np.linalg.norm(cross23[1, :] - np.array([x1, y1]))
        if d1 < d2:
            threePoints[1, :] = cross23[0, :]
        else:
            threePoints[1, :] = cross23[1, :]

    if len(cross13) == 2:
        d1 = np.linalg.norm(cross13[0, :] - np.array([x2, y2]))
        d2 = np.linalg.norm(cross13[1, :] - np.array([x2, y2]))
        if d1 < d2:
            threePoints[2, :] = cross13[0, :]
        else:
            threePoints[2, :] = cross13[1, :]
    return threePoints


# 绘制定位示意图
def drawfigure(x123, y123, r123, three, realxy, estixy, filename):
    plt.close()
    alpha = np.arange(0, 2 * np.pi, 0.001)
    x = x123[0] + r123[0] * np.cos(alpha)
    y = y123[0] + r123[0] * np.sin(alpha)
    plt.plot(x, y, linewidth=0.5)

    x = x123[1] + r123[1] * np.cos(alpha)
    y = y123[1] + r123[1] * np.sin(alpha)
    plt.plot(x, y, linewidth=0.5)

    x = x123[2] + r123[2] * np.cos(alpha)
    y = y123[2] + r123[2] * np.sin(alpha)
    plt.plot(x, y, linewidth=0.5)

    plt.plot(x123[0], y123[0], 'ro', markersize=1)
    plt.plot(x123[1], y123[1], 'ro', markersize=1)
    plt.plot(x123[2], y123[2], 'ro', markersize=1)
    plt.plot(realxy[0], realxy[1], 'ro', markersize=2)
    if len(three) == 3:
        plt.plot(three[:, 0], three[:, 1], 'k^', markersize=2)

    plt.plot([x123[0], x123[1]], [y123[0], y123[1]], linewidth=0.5, c='r', ls='-')
    plt.plot([x123[0], x123[2]], [y123[0], y123[2]], linewidth=0.5, c='r', ls='-')
    plt.plot([x123[1], x123[2]], [y123[1], y123[2]], linewidth=0.5, c='r', ls='-')

    plt.text(x123[0], y123[0], "wrd", fontsize=9, color="k", style="italic", weight="light",
             verticalalignment='bottom',
             horizontalalignment='left')
    plt.text(x123[1], y123[1], "zmz", fontsize=9, color="k", style="italic", weight="light",
             verticalalignment='bottom',
             horizontalalignment='left')
    plt.text(x123[2], y123[2], "zy", fontsize=9, color="k", style="italic", weight="light",
             verticalalignment='bottom',
             horizontalalignment='left')
    plt.text(realxy[0], realxy[1], "r", fontsize=9, color="k", style="italic", weight="light",
             verticalalignment='top',
             horizontalalignment='right')

    if not (estixy == np.array([0, 0])).all():
        plt.plot(estixy[0], estixy[1], 'b*', markersize=3)
        plt.text(estixy[0], estixy[1], "e", fontsize=9, color="k", style="italic", weight="light",
                 verticalalignment='top',
                 horizontalalignment='right')
    plt.axis('scaled')
    plt.savefig(filename, dpi=1080, bbox_inches='tight')
    # plt.show()
    # return plt
