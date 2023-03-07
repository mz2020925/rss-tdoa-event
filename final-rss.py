import math
import numpy as np


def fiveTothree(dBuv1: list, dBuv2: list, dBuv3: list, dBuv4: list, dBuv5: list, receivegps: list) -> list:
    """
    本函数从 5个接收机的场强数据和经纬度 优选出 3个接收机的场强数据和经纬度
    :param dBuv1:第一台接收机的接收场强列表（后端控制RSA306B读取）
    :param dBuv2:
    :param dBuv3:
    :param dBuv4:
    :param dBuv5:
    :param receivegps: list of list或tuple -> [[经度，维度],[],[],[],[]] 或 [(经度，维度),(),(),(),()]
    :return: list of dict -> [{"rss": rss1, "gps": receivegps[0]},{},{}]

    *注意 receivegps中的 5台接收机gps要顺序对应着 dBuv1, dBuv2, dBuv3, dBuv4, dBuv5，
    因此在本函数体中构建字典 point1、point2、point3、point4、point5
    """
    # 从每个接收机数据中提取最大num=6组接收功率
    num = 6
    sortData = np.sort(dBuv1)  # 默认升序排列
    rss1 = np.mean(sortData[len(sortData) - num:])  # dBm
    sortData = np.sort(dBuv2)
    rss2 = np.mean(sortData[len(sortData) - num:])  # dBm
    sortData = np.sort(dBuv3)
    rss3 = np.mean(sortData[len(sortData) - num:])  # dBm
    sortData = np.sort(dBuv4)
    rss4 = np.mean(sortData[len(sortData) - num:])  # dBm
    sortData = np.sort(dBuv5)
    rss5 = np.mean(sortData[len(sortData) - num:])  # dBm

    point1 = {"rss": rss1, "gps": receivegps[0]}
    point2 = {"rss": rss2, "gps": receivegps[1]}
    point3 = {"rss": rss3, "gps": receivegps[2]}
    point4 = {"rss": rss4, "gps": receivegps[3]}
    point5 = {"rss": rss5, "gps": receivegps[4]}

    templist = [point1, point2, point3, point4, point5]
    index = np.argsort([point1["rss"], point2["rss"], point3["rss"], point4["rss"], point5["rss"]])
    max3points = [templist[int(index[-1])], templist[int(index[-2])], templist[int(index[-3])]]

    return max3points


def rss_method(point1: dict, point2: dict, point3: dict, senddBm=-7) -> np.ndarray:  # 发射场强设置假设为100时，发射功率即为-7
    """
    本函数利用 3个接收机的场强数据 确定出 三角形模糊区域的三个顶点坐标
    :param point1:{'rss': [ , , ...], 'gps': [ , ]}
    :param point2:
    :param point3:
    :param senddBm:
    :return:
    """
    # 算出功率差即是信号传输损耗
    less1 = senddBm - point1["rss"]
    less2 = senddBm - point2["rss"]
    less3 = senddBm - point3["rss"]

    # 根据拟合出的n，估算出信号源到接收机的距离
    n1 = -0.02972 * point1["rss"] + 0.854
    r1arr = np.power(10, less1 / (10 * n1))
    r1 = np.mean(r1arr)

    n2 = -0.02972 * point2["rss"] + 0.854
    r2arr = np.power(10, less2 / (10 * n2))
    r2 = np.mean(r2arr)

    n3 = -0.02972 * point3["rss"] + 0.854
    r3arr = np.power(10, less3 / (10 * n3))
    r3 = np.mean(r3arr)
    # print(f'原始半径:r1={r1},r2={r2},r3={r3}')

    # 提取接收机经纬度
    long1 = point1["gps"][0]
    lati1 = point1["gps"][1]

    long2 = point2["gps"][0]
    lati2 = point2["gps"][1]

    long3 = point3["gps"][0]
    lati3 = point3["gps"][1]

    # 建立相对坐标系，以接收机点(long1,lati1)作为参考原点(0,0)，计算接收机相对坐标
    x1, y1 = lonlat2xy(long1, lati1, long1, lati1)
    x2, y2 = lonlat2xy(long2, lati2, long1, lati1)
    x3, y3 = lonlat2xy(long3, lati3, long1, lati1)

    # 计算接收机构成的三角形的边长
    d12 = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
    d23 = np.linalg.norm(np.array([x2, y2]) - np.array([x3, y3]))
    d13 = np.linalg.norm(np.array([x1, y1]) - np.array([x3, y3]))

    # 设置三个半径的上限
    if r1 > np.max(np.array([d12, d13])):
        r1 = np.max(np.array([d12, d13]))
    if r2 > np.max(np.array([d12, d23])):
        r2 = np.max(np.array([d12, d23]))
    if r3 > np.max(np.array([d23, d13])):
        r3 = np.max(np.array([d23, d13]))

    # 计算三角形的三个顶点坐标
    cross12 = circleCross(x1, y1, r1, x2, y2, r2, d12)
    cross23 = circleCross(x2, y2, r2, x3, y3, r3, d23)
    cross13 = circleCross(x1, y1, r1, x3, y3, r3, d13)
    three = threeCrosspoint(x1, y1, x2, y2, x3, y3, cross12, cross23, cross13)

    # 测试定位误差--加权质心法定位
    # estix = (three[0, 0] / (r1 + r2) + three[1, 0] / (r2 + r3) + three[2, 0] / (r1 + r3)) / (
    #         1 / (r1 + r2) + 1 / (r2 + r3) + 1 / (r1 + r3))
    # estiy = (three[0, 1] / (r1 + r2) + three[1, 1] / (r2 + r3) + three[2, 1] / (r1 + r3)) / (
    #         1 / (r1 + r2) + 1 / (r2 + r3) + 1 / (r1 + r3))
    # estixy = np.array([estix, estiy])
    # realgps = [103.92736173333333, 30.752247883333336]
    # realxy = np.array(lonlat2xy(realgps[0], realgps[1], long1, lati1))
    # distance = np.linalg.norm(estixy - realxy)
    # print(f'估计坐标与实际坐标的距离{distance}')

    return three


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


def circleCross(x0, y0, r0, x1, y1, r1, distance):
    """
    求两相交圆的交点，返回两个交点坐标
    :param x0:
    :param y0:
    :param r0:
    :param x1:
    :param y1:
    :param r1:
    :param distance:
    :return:
    """
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


def threeCrosspoint(x1, y1, x2, y2, x3, y3, cross12, cross23, cross13):
    """
    求出三圆相交时所需的三个点的坐标
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param x3:
    :param y3:
    :param cross12:
    :param cross23:
    :param cross13:
    :return:
    """
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


def judge_triangle(x1, y1, x2, y2, x3, y3, x, y):
    """
    此函数判断给定坐标点是否在三角形内部
    :param x1: 三角形顶点1，x坐标
    :param y1: 三角形顶点1，y坐标
    :param x2:
    :param y2:
    :param x3:
    :param y3:
    :param x: 待判断的点的x坐标
    :param y: 待判断的点的y坐标
    :return:
    """
    A = np.array([x1, y1, 0])
    B = np.array([x2, y2, 0])
    C = np.array([x3, y3, 0])
    P = np.array([x, y, 0])

    AB = A - B
    AP = A - P
    AC = A - C

    BA = B - A
    BP = B - P
    BC = B - C

    BABP = np.cross(BA, BP)
    BPBC = np.cross(BP, BC)
    ABAP = np.cross(AB, AP)
    APAC = np.cross(AP, AC)

    if BABP[2] * BPBC[2] > 0 and ABAP[2] * APAC[2] > 0:
        return True  # 在三角形内部
    else:
        return False  # 不在三角形内部


if __name__ == '__main__':
    # 给定测试数据point4.mat
    dBuv1 = [-49.72459923,
             -49.41994063,
             -49.02202597,
             -48.94452902,
             -49.32255906,
             -49.91765407,
             -49.61548067,
             -49.52517543,
             -48.5246788,
             -48.08772503]
    dBuv2 = [-30.1227372,
             -29.94373459,
             -29.61602039,
             -29.52870874,
             -31.12071049,
             -26.38113252,
             -25.70460909,
             -26.23058483,
             -26.12678674,
             -25.57604828]
    dBuv3 = [-55.25980623,
             -55.8809778,
             -56.22607418,
             -55.82873058,
             -55.76535261,
             -55.51958544,
             -55.46310109,
             -55.52197689,
             -55.41390325,
             -56.77402129]
    dBuv4 = [-55.25980623,
             -55.8809778,
             -56.22607418,
             -55.82873058,
             -55.76535261,
             -55.51958544,
             -55.46310109,
             -55.52197689,
             -55.41390325,
             -56.77402129]
    dBuv5 = [-55.25980623,
             -55.8809778,
             -56.22607418,
             -55.82873058,
             -55.76535261,
             -55.51958544,
             -55.46310109,
             -55.52197689,
             -55.41390325,
             -56.77402129]
    receivegps = [[103.92781393333334, 30.752267400000004], [103.92724926666668, 30.7524481],
                  [103.92741038333334, 30.7517394], [103.92741038333334, 30.7517394], [103.92741038333334, 30.7517394]]
    print("完成给定测试数据；")

    # 1.采集5个接收机的场强数据和gps数据
    # dBuv1 = []
    # dBuv2 = []
    # dBuv3 = []
    # dBuv4 = []
    # dBuv5 = []
    # receivegps = [[], [], [], [], []]

    # 2.通过下面两个函数计算出三角形模糊区域的三个顶点坐标
    max3points = fiveTothree(dBuv1, dBuv2, dBuv3, dBuv4, dBuv5, receivegps)  # 注意函数参数receivegps的格式
    three = rss_method(max3points[0], max3points[1], max3points[2],
                       senddBm=-7)  # three的定义为：three = np.array([[0., 0.], [0., 0.], [0., 0.]])
    # print(three)

    # 3.在三角形模糊区域内利用反射性匹配法
    x = 0.  # 网格坐标
    y = 0.
    flag = judge_triangle(x1=three[0][0], y1=three[0][1],
                          x2=three[1][0], y2=three[1][1],
                          x3=three[2][0], y3=three[2][1],
                          x=x, y=y)
