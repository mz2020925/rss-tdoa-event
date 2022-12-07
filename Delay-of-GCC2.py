import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os


def timeDelay(data1, data2, subFs=1e8):  # data1, data2长度一样
    # print(len(data1), len(data2))
    autocorr = signal.fftconvolve(data1, data2, mode='full')
    # autocorr = np.abs(autocorr)
    print(len(autocorr))
    # np.argmax(autocorr获取的是在ndarray中的所以，不是互相关实际的平移点数，要减去(len(data1) + len(data2) - 1 - 1) / 2.0
    result = autocorr[int((len(autocorr) - 1) / 2) - 1024: int((len(autocorr) - 1) / 2 + 1024) + 1]
    x = np.linspace(0, len(result), len(result))
    plt.scatter(x, result, marker='.', linewidths=0.1)
    plt.show()
    offset = np.argmax(result) - int((len(result) - 1) / 2)
    print(offset)

    timedelay = 1.0 / subFs * offset
    # print(f'做互相关前输入数据长度：{len(data1)}')
    # print(f'做互相关后输出数据长度：{len(autocorr)}')
    print(f'相关性最大值对应的移位：{offset}')
    return timedelay


def calculateTimeDelay(I1, Q1, I2, Q2, currFs=256000, start=0, end_no=4096, subFs=1e8):
    # subFs = 1e8  # 重采样的频率
    # 重采样
    I1_resample = signal.resample_poly(I1[start:end_no], np.ceil((end_no - start - 1) * subFs / currFs) + 1,
                                       end_no - start, padtype='mean', window=signal.windows.kaiser(4096, beta=14))
    Q1_resample = signal.resample_poly(Q1[start:end_no], np.ceil((end_no - start - 1) * subFs / currFs) + 1,
                                       end_no - start, padtype='mean', window=signal.windows.kaiser(4096, beta=14))
    I2_resample = signal.resample_poly(I2[start:end_no], np.ceil((end_no - start - 1) * subFs / currFs) + 1,
                                       end_no - start, padtype='mean', window=signal.windows.kaiser(4096, beta=14))
    Q2_resample = signal.resample_poly(Q2[start:end_no], np.ceil((end_no - start - 1) * subFs / currFs) + 1,
                                       end_no - start, padtype='mean', window=signal.windows.kaiser(4096, beta=14))

    # 扩大信号幅度
    # increase = 1e4
    I1_resample = (I1_resample - np.min(I1_resample)) / (np.max(I1_resample) - np.min(I1_resample))
    Q1_resample = (Q1_resample - np.min(Q1_resample)) / (np.max(Q1_resample) - np.min(Q1_resample))
    I2_resample = (I2_resample - np.min(I2_resample)) / (np.max(I2_resample) - np.min(I2_resample))
    Q2_resample = (Q2_resample - np.min(Q2_resample)) / (np.max(Q2_resample) - np.min(Q2_resample))

    timedelay12 = np.min([timeDelay(I1_resample, I2_resample, subFs), timeDelay(Q1_resample, Q2_resample, subFs)])
    # timedelay13 = np.min(timeDelay(I1_resample, I3_resample, subFs), timeDelay(Q1_resample, Q3_resample, subFs))
    return timedelay12


def main():
    fileDir = r"E:\Event1\师姐给的1月14号\IQ数据（TDOA）\cdz测试数据20220518"
    dirList = os.listdir(fileDir)  # ['2188', '2587', '2775', '文件名说用.txt']
    os.chdir(fileDir + "\\" + dirList[1])
    # print(os.getcwd())
    fileList = os.listdir(os.getcwd())
    # print(fileList)

    I = np.zeros(shape=(4196, 3))
    Q = np.zeros(shape=(4196, 3))
    for i in range(len(fileList) - 1):
        with open(fileList[i], encoding='utf-8') as f:
            # print(fileList[i])
            data = np.loadtxt(fileList[i], dtype=float, skiprows=1, delimiter='\t')
            I[:, i] = np.array([data[:, 0]])
            Q[:, i] = np.array([data[:, 1]])
    # print(I[-5:, :])
    # print(Q[-5:, :])
    print("完成数据读取；")
    # x = np.linspace(0, 499, 499)
    # y = I[:499, 0]
    # plt.subplot(311)
    # plt.plot(x, y, linestyle='solid', color='blue')
    # plt.subplot(312)
    # y = I[:499, 1]
    # plt.plot(x, y, linestyle='solid', color='blue')
    # plt.subplot(313)
    # y = I[:499, 2]
    # plt.plot(x, y, linestyle='solid', color='blue')
    # plt.show()

    timedelay12 = calculateTimeDelay(I[:, 0], Q[:, 0], I[:, 1], Q[:, 1], subFs=1e8)
    # timedelay13 = calculateTimeDelay(I[:, 0], Q[:, 0], I[:, 2], Q[:, 2])
    # print(f'时间差1-2：{timedelay12}ns, 时间差1-3：{timedelay13}ns')


if __name__ == '__main__':
    main()
