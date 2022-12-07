import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def timeDelay(data1, data2, ff):  # data1, data2长度一样
    # print(len(data1), len(data2))
    autocorr = signal.fftconvolve(data1, data2, mode='full')

    # result = autocorr[int((len(autocorr) - 1) / 2) - 1024: int((len(autocorr) - 1) / 2 + 1024) + 1]
    # x = np.linspace(0, len(autocorr), len(autocorr))
    # plt.scatter(x, autocorr, marker='.', linewidths=0.1)
    # plt.show()
    # np.argmax(autocorr获取的是在ndarray中的所以，不是互相关实际的平移点数，要减去(len(data1) + len(data2) - 1 - 1) / 2.0
    index = np.argmax(autocorr) - (len(data1) + len(data2) - 1 - 1) / 2.0
    timedelay = 1.0 / ff * index * 1.0e9
    # print(f'做互相关前输入数据长度：{len(data1)}')
    # print(f'做互相关后输出数据长度：{len(autocorr)}')
    print(f'相关性最大值对应的移位：{index}')
    return timedelay


def GCC_get_Timedelay(I, Q, n=4096, f0=256000, ff=100000000):  # 给的数据I、Q分别都有4096个点;实际采样频率；重采样频率
    # 计算出IQ数据的平方和
    # trace1 = np.power(I[:, 0], 2) + np.power(Q[:, 0], 2)
    # power1 = 10*np.log10(trace1*1000)
    # std1 = (power1-np.min(power1))/(np.max(power1)-np.min(power1))
    # trace2 = np.power(I[:, 1], 2) + np.power(Q[:, 1], 2)
    # trace3 = np.power(I[:, 2], 2) + np.power(Q[:, 2], 2)

    # 重采样
    resampleI = np.zeros(shape=(int(np.ceil(n * ff / f0)), 3))
    resampleQ = np.zeros(shape=(int(np.ceil(n * ff / f0)), 3))
    # n_new = np.linspace(0, np.ceil(n * ff / f0), int(np.ceil(n * ff / f0)), endpoint=False)
    for i in range(3):
        resampleI[:, i] = signal.resample(I[:, i], int(np.ceil(n * ff / f0)))
        resampleQ[:, i] = signal.resample(Q[:, i], int(np.ceil(n * ff / f0)))
    print("完成重采样；")
    x = np.linspace(0, 499999,499999)
    y = resampleI[:499999, 0]
    plt.subplot(311)
    plt.plot(x, y, linestyle='solid', color='blue')
    plt.subplot(312)
    y = resampleI[:499999, 1]
    plt.plot(x, y, linestyle='solid', color='blue')
    plt.subplot(313)
    y = resampleI[:499999, 2]
    plt.plot(x, y, linestyle='solid', color='blue')
    plt.show()
    # I1_poly = signal.resample_poly(I[:, 0], np.ceil(n * ff / f0), 4096)
    # I2_poly = signal.resample_poly(I[:, 1], np.ceil(n * ff / f0), 4096)
    # I3_poly = signal.resample_poly(I[:, 2], np.ceil(n * ff / f0), 4096)
    #
    # Q1_poly = signal.resample_poly(Q[:, 0], np.ceil(n * ff / f0), 4096)
    # Q2_poly = signal.resample_poly(Q[:, 1], np.ceil(n * ff / f0), 4096)
    # Q3_poly = signal.resample_poly(Q[:, 2], np.ceil(n * ff / f0), 4096)

    # 归一化处理
    for i in range(3):
        resampleI[:, i] = (resampleI[:, i] - np.min(resampleI[:, i])) / (
                np.max(resampleI[:, i]) - np.min(resampleI[:, i]))
        resampleQ[:, i] = (resampleQ[:, i] - np.min(resampleQ[:, i])) / (
                np.max(resampleQ[:, i]) - np.min(resampleQ[:, i]))
    print("完成归一化处理；")
    # stdI1 = (I[:, 0] - np.min(I[:, 0])) / (np.max(I[:, 0]) - np.min(I[:, 0]))
    # stdI2 = (I[:, 1] - np.min(I[:, 1])) / (np.max(I[:, 1]) - np.min(I[:, 1]))
    # stdI3 = (I[:, 2] - np.min(I[:, 2])) / (np.max(I[:, 2]) - np.min(I[:, 2]))
    #
    # stdQ1 = (Q[:, 0] - np.min(Q[:, 0])) / (np.max(Q[:, 0]) - np.min(Q[:, 0]))
    # stdQ2 = (Q[:, 1] - np.min(Q[:, 1])) / (np.max(Q[:, 1]) - np.min(Q[:, 1]))
    # stdQ3 = (Q[:, 2] - np.min(Q[:, 2])) / (np.max(Q[:, 2]) - np.min(Q[:, 2]))

    timedelay12 = (timeDelay(resampleI[:, 0], resampleI[:, 1], ff) + timeDelay(resampleQ[:, 0], resampleQ[:, 1],
                                                                               ff)) / 2.0
    timedelay13 = (timeDelay(resampleI[:, 0], resampleI[:, 2], ff) + timeDelay(resampleQ[:, 0], resampleQ[:, 2],
                                                                               ff)) / 2.0
    print("完成时间差计算；")
    return timedelay12, timedelay13


def main():
    fileDir = r"E:\Event1\师姐给的1月14号\IQ数据（TDOA）\cdz测试数据20220518"
    dirList = os.listdir(fileDir)  # ['2188', '2587', '2775', '文件名说用.txt']
    os.chdir(fileDir + "\\" + dirList[0])
    # print(os.getcwd())
    fileList = os.listdir(os.getcwd())
    # print(fileList)

    I = np.zeros(shape=(4096, 3))
    Q = np.zeros(shape=(4096, 3))
    for i in range(len(fileList) - 1):
        with open(fileList[i], encoding='utf-8') as f:
            # print(fileList[i])
            data = np.loadtxt(fileList[i], dtype=float, skiprows=1, delimiter='\t')
            I[:, i] = np.array([data[:, 0]])
            Q[:, i] = np.array([data[:, 1]])
    # print(I[-5:, :])
    # print(Q[-5:, :])
    print("完成数据读取；")
    x = np.linspace(0, 499,499)
    y = I[:499, 0]
    plt.subplot(311)
    plt.plot(x, y, linestyle='solid', color='blue')
    plt.subplot(312)
    y = I[:499, 1]
    plt.plot(x, y, linestyle='solid', color='blue')
    plt.subplot(313)
    y = I[:499, 2]
    plt.plot(x, y, linestyle='solid', color='blue')
    plt.show()

    timedelay12, timedelay13 = GCC_get_Timedelay(I, Q, ff=100000000)
    print(f'时间差1-2：{timedelay12}ns, 时间差1-3：{timedelay13}ns')


if __name__ == '__main__':
    main()
