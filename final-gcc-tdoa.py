import numpy as np
import matplotlib.pyplot as plt
import os
import math


def time_delay(data1, data2, fs=1280000):
    """
    本函数根据IQ数据计算互相关，从而得到时间差
    :param data1: list
    :param data2: list
    :param fs: 采样率
    :return: 时间差
    """
    fun = lambda x: x / np.mean(x) - 1
    data1 = fun(data1)
    data2 = fun(data2)
    corr_array = np.correlate(data1, data2, 'full')
    index1 = np.argmax(abs(corr_array))
    length = np.size(corr_array)
    center = (length - 1) / 2
    delay = (index1 - center) * (1 / fs)

    # 画出图像
    # x = np.linspace(0, np.size(corr_array), np.size(corr_array))
    # plt.scatter(x, corr_array, linewidths=0.1, marker='.')
    # plt.show()
    return delay


def tdoa3(long1, lati1, long2, lati2, long3, lati3, delay21, delay31):
    """
    使用3基站chan算法估算位置
    :param long1:
    :param lati1:
    :param long2:
    :param lati2:
    :param long3:
    :param lati3:
    :param delay21:
    :param delay31:
    :return:
    """
    # 建立相对坐标系，以接收机点(long1,lati1)作为参考原点(0,0)，计算接收机相对坐标
    x1, y1 = lonlat2xy(long1, lati1, long1, lati1)
    x2, y2 = lonlat2xy(long2, lati2, long1, lati1)
    x3, y3 = lonlat2xy(long3, lati3, long1, lati1)
    # 打印信号接收机之间的距离
    # 略

    # 将(x1, y1)作为TDOA算法中的参考基站
    x21 = x2 - x1
    x31 = x3 - x1

    y21 = y2 - y1
    y31 = y3 - y1

    r21 = delay21 * 3e8  # 这里就用到了TDOA, r21 = r2 - r1 = (toa2-toa1) * c = toda21 * c
    r31 = delay31 * 3e8  # 这里就用到了TDOA

    K1 = x1 ** 2 + y1 ** 2
    K2 = x2 ** 2 + y2 ** 2
    K3 = x3 ** 2 + y3 ** 2

    p1 = (y21 * r31 ** 2 - y31 * r21 ** 2 + y31 * (K2 - K1) - y21 * (K3 - K1)) / (2 * (x21 * y31 - x31 * y21))
    p2 = (x21 * r31 ** 2 - x31 * r21 ** 2 + x31 * (K2 - K1) - x21 * (K3 - K1)) / (2 * (x31 * y21 - x21 * y31))
    q1 = (y21 * r31 - y31 * r21) / (x21 * y31 - x31 * y21)
    q2 = (x21 * r31 - x31 * r21) / (x31 * y21 - x21 * y31)

    a = q1 ** 2 + q2 ** 2 - 1
    b = -2 * (q1 * (x1 - p1) + q2 * (y1 - p2))
    c = (x1 - p1) ** 2 + (y1 - p2) ** 2

    # 求解一元二次方程组
    res_roots = np.roots([a, b, c])
    print(f'一元二次方程的根{res_roots}')

    # res_roots计算出来具有2个值是数学意义，一方面R1有物理意义，负值应该舍去，
    # 但是另一方面，有时候会出现res_roots都是正值的情况，这时候需要设定定位的范围，范围之外的解不要。
    # 本例中res_roots都是正值的情况，且定位结果相近。
    estixy = []
    for item in res_roots:
        if abs(item) > 0:
            estixy.append([p1 + q1 * item, p2 + q2 * item])

    # 测试定位误差
    realgps = [103.92736173333333, 30.752247883333336]
    realxy = np.array(lonlat2xy(realgps[0], realgps[1], long1, lati1))
    calculate_xy1 = np.array(estixy[0])
    calculate_xy2 = np.array(estixy[1])
    distance1 = np.linalg.norm(calculate_xy1 - realxy)
    distance2 = np.linalg.norm(calculate_xy2 - realxy)
    print(f'测试3基站定位误差{distance1, distance2}')

    return estixy


def tdoa5(long1, lati1, long2, lati2, long3, lati3, long4, lati4, long5, lati5, BSN, R):
    # , long4, lati4, long5, lati5
    x1, y1 = lonlat2xy(long1, lati1, long1, lati1)
    x2, y2 = lonlat2xy(long2, lati2, long1, lati1)
    x3, y3 = lonlat2xy(long3, lati3, long1, lati1)
    x4, y4 = lonlat2xy(long4, lati4, long1, lati1)
    x5, y5 = lonlat2xy(long5, lati5, long1, lati1)
    BS = [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]]
    # BS = [[x1, y1], [x2, y2], [x3, y3]]
    # BSN为基站数目,BS为各基站坐标矩阵(BSN*2)(BS第一行是参考基站的坐标)(墨卡托坐标)，
    # R为基站记录距离差数组，长度为BSN-1，分别为各基站到参考基站的坐标
    # 噪声协方差矩阵
    Q = np.eye(BSN - 1)
    # 第一次LS
    # K
    K1 = 0
    K = np.zeros(BSN)
    for i in range(BSN - 1):
        K[i + 1] = np.power(BS[i + 1][0], 2) + np.power(BS[i + 1][1], 2)
    # Ga
    Ga = np.zeros((BSN - 1, 3))
    for i in range(BSN - 1):
        Ga[i][0] = -BS[i + 1][0]
        Ga[i][1] = -BS[i + 1][1]
        Ga[i][2] = -R[i]
    # h
    h = np.zeros(BSN - 1)
    for i in range(BSN - 1):
        h[i] = 0.5 * (np.power(R[i], 2) - K[i + 1] + K[0])

    # 给出Z的初始值
    Za0 = np.linalg.inv(Ga.T @ np.linalg.inv(Q) @ Ga) @ Ga.T @ np.linalg.inv(Q) @ h.T
    # 利用这个粗略估计值计算B
    B = np.eye(BSN - 1)
    for i in range(BSN - 1):
        B[i][i] = np.sqrt(np.power(BS[i + 1][0] - Za0[0], 2) + np.power(BS[i + 1][1] - Za0[1], 2))
    # FI
    FI = B @ Q @ B
    # 第一次LS结果
    Za1 = np.linalg.inv(Ga.T @ np.linalg.inv(FI) @ Ga) @ Ga.T @ np.linalg.inv(FI) @ h.T

    # 第二次WLS
    # 第一次LS结果的协方差
    CovZa = np.linalg.inv(Ga.T @ np.linalg.inv(FI) @ Ga)
    # 第二次LS
    sB = np.eye(3)
    for i in range(3):
        sB[i][i] = Za1[i]
    # sFI
    sFI = 4 * sB @ CovZa @ sB
    # sGa
    sGa = np.array([[1, 0],
                    [0, 1],
                    [1, 1]])
    sh = np.array([[np.power(Za1[0], 2)],
                   [np.power(Za1[1], 2)],
                   [np.power(Za1[2], 2)]])
    # 第二次LS结果
    Za2 = np.linalg.inv(sGa.T @ np.linalg.inv(sFI) @ sGa) @ sGa.T @ np.linalg.inv(sFI) @ sh
    sZ = np.sqrt(abs(Za2))

    # 测试定位误差
    # realgps = [103.92736173333333, 30.752247883333336]
    # realxy = np.array(lonlat2xy(realgps[0], realgps[1], long1, lati1))
    # calculate_xy1 = np.array([sZ[0][0], sZ[1][0]])
    # distance1 = np.linalg.norm(calculate_xy1 - realxy)
    # print(f'测试多基站定位误差{distance1}')

    # 输出
    return sZ


def lonlat2xy(lon, lat, ref_lon, ref_lat):
    """
    经纬度转相对xy
    :param lon:
    :param lat:
    :param ref_lon:
    :param ref_lat:
    :return:
    """
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


def xy2lonlat(x, y, ref_lon, ref_lat):
    """
    相对xy转经纬度
    :param x:
    :param y:
    :param ref_lon:
    :param ref_lat:
    :return:
    """
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


def judge_circle(estix, estiy, r0, searchx, searchy):
    """
    :param estix:
    :param estiy:
    :param r0:
    :param searchx:
    :param searchy:
    :return:
    """
    estixy = np.array([estix, estiy])
    searchxy = np.array([searchx, searchy])
    distance = np.linalg.norm(estixy - searchxy)

    if r0 > distance:
        return True  # 在圆形内部
    else:
        return False  # 不在圆形内部


if __name__ == '__main__':
    # 给定测试数据
    fileDir = r"E:\Event-大运会定位\师姐给的1月14号\IQ数据（TDOA）\cdz测试数据20220518"
    dirList = os.listdir(fileDir)  # ['2188', '2587', '2775', '文件名说用.txt']
    os.chdir(fileDir + "\\" + dirList[1])  # dirList[1]是文件夹'2587'
    # print(os.getcwd())
    fileList = os.listdir(os.getcwd())  # fileList=[.xls, .xls, .xls, .txt]
    # print(fileList)
    I = np.zeros(shape=(4196, 3))
    Q = np.zeros(shape=(4196, 3))
    for i in range(len(fileList) - 1):  # 减1是不打开.txt文件
        with open(fileList[i], encoding='utf-8') as f:
            # print(fileList[i])
            data = np.loadtxt(fileList[i], dtype=float, skiprows=1, delimiter='\t')
            I[:, i] = np.array([data[:, 0]])
            Q[:, i] = np.array([data[:, 1]])
    # 数据分别存入不同的数组中
    I1 = I[:, 0]
    I2 = I[:, 1]
    I3 = I[:, 2]
    Q1 = Q[:, 0]
    Q2 = Q[:, 1]
    Q3 = Q[:, 2]

    # 接收机1、2、3、4、5的经纬度（由于测试只有3个接收机，3、4、5接收机的经纬度相同）
    receivegps = [[103.92781393333334, 30.752267400000004], [103.92724926666668, 30.7524481],
                  [103.92741038333334, 30.7517394]]
    # print("完成给定测试数据；")

    # 1.采集5个接收机的场强数据和gps数据
    dBuv1 = []
    dBuv2 = []
    dBuv3 = []
    dBuv4 = []
    dBuv5 = []
    receivegps = [[], [], [], [], []]

    # 2.通过下面两个函数计算出时间差和给出圆形模糊区域
    fs_max = 40000000  # 最大采样率，到时候采样率就设置这个数
    fs = 1280000  # 这是测试数据中的采样率
    time_delay12 = (time_delay(I1, I2, fs) + time_delay(Q1, Q2, fs)) / 2
    time_delay13 = (time_delay(I1, I3, fs) + time_delay(Q1, Q3, fs)) / 2
    time_delay14 = (time_delay(I1, I4, fs) + time_delay(Q1, Q4, fs)) / 2
    time_delay15 = (time_delay(I1, I5, fs) + time_delay(Q1, Q5, fs)) / 2

    # estixy = tdoa3(long1=receivegps[0][0], lati1=receivegps[0][1],
    #                long2=receivegps[1][0], lati2=receivegps[1][1],
    #                long3=receivegps[2][0], lati3=receivegps[2][1],
    #                delay21=time_delay12, delay31=time_delay13)
    # print(f'3基站定位结果\n{estixy}')

    BSN = 5
    R = [time_delay12, time_delay13, time_delay14, time_delay15]
    estixy = tdoa5(long1=receivegps[0][0], lati1=receivegps[0][1],
                   long2=receivegps[1][0], lati2=receivegps[1][1],
                   long3=receivegps[2][0], lati3=receivegps[2][1],
                   long4=receivegps[2][0], lati4=receivegps[2][1],
                   long5=receivegps[2][0], lati5=receivegps[2][1],
                   BSN=BSN, R=R)
    # print(f'多基站定位结果\n{estixy}')

    # 3.在圆形形模糊区域内利用反射性匹配法
    # r0 = 0.
    # searchx = 0.
    # searchy = 0.
    # flag = judge_circle(estix=estixy[0], estiy=estixy[1], r0=r0, searchx=searchx, searchy=searchy)
