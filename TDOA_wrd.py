import numpy as np
import math


def two_wls(BSN, BS, R):
    # BSN为基站数目,BS为各基站坐标矩阵(BSN*2)(BS第一行为参考基站的坐标)(墨卡托坐标)，
    # R为基站记录距离差数组，长度为BSN-1，分别为各基站到参考基站的坐标
    Q = np.eye(BSN - 1)
    B = np.eye(BSN - 1)
    K = []
    h = []
    Ga = np.zeros((BSN - 1, 3))
    for i in range(BSN):
        K[i] = pow(BS[i][1], 2) + pow(BS[i][2], 2)
    for i in range(BSN - 1):
        Ga[i][1] = -(BS[i + 1][1] - BS[1][1])
        Ga[i][2] = -(BS[i + 1][2] - BS[1][2])
        Ga[i][3] = -R(i)
    for i in range(BSN - 1):
        h[i] = 0.5 * (R[i] ^ 2 - K[i+1] - K[0])
    Q_inv = np.linalg.inv(Q)
    Z0_1 = np.dot(np.linalg.inv(np.dot(np.dot(Ga.T, Q_inv), Ga)), Ga.T)
    Z0_2 = np.dot(Q_inv, h.T)
    Z0 = np.dot(Z0_1, Z0_2)
    for i in range(BSN - 1):
        B[i][i] = math.sqrt(pow(BS[i+1][1] - Z0[1], 2) + pow(BS[i+1][2] - Z0[2], 2))
    FI = np.dot(np.dot(B, Q), B)
    FI_inv = np.linalg.inv(FI)
    result_1_1 = np.dot(np.linalg.inv(np.dot(np.dot(Ga.T, FI_inv), Ga)), Ga.T)
    result_1_2 = np.dot(FI_inv, h.T)

    # 第一次WLS结果
    result_1 = np.dot(result_1_1, result_1_2)

    # 第一次LS结果的方差
    Cov_result_1 = np.linalg.inv(np.dot(np.dot(Ga.T, FI_inv), Ga))

    # 第二次LS
    sB = np.eye(3)
    for i in range(3):
        sB[i][i] = result_1[i]
    sFI = 4*np.dot(np.dot(sB, Cov_result_1), sB)
    sGa = np.array([[1, 0], [0, 1], [1, 1]])
    sh = np.array([[pow(result_1[0], 2)], [pow(result_1[1], 2)], [pow(result_1[2], 2)]])

    # 第二次WLS结果
    sFI_inv = np.linalg.inv(sFI)
    result_2_1 = np.linalg.inv(np.dot(np.dot(sGa.T, sFI_inv), sGa))
    result_2_1 = np.dot(result_2_1, sGa.T)
    result_2_2 = np.dot(sFI_inv, sh)
    result_2 = np.dot(result_2_1, result_2_2)
    sZ = math.sqrt(abs(result_2))
    return sZ


