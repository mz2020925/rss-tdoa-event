import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import os


def main():
    fileDir = r"E:\Event-大运会定位\师姐给的1月14号\IQ数据（TDOA）\cdz测试数据20220518"
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
    print("完成数据读取；")

    # 数据分解
    I1 = I[:, 0]
    I2 = I[:, 1]
    I3 = I[:, 2]
    Q1 = Q[:, 0]
    Q2 = Q[:, 1]
    Q3 = Q[:, 2]

    # 数据合成复数
    IQ1 = I1 + 1j * Q1
    IQ1 = IQ1-np.mean(IQ1)
    IQ2 = I2 + 1j * Q2
    IQ2 = IQ2 - np.mean(IQ2)
    IQ3 = I3 + 1j * Q3
    IQ3 = IQ3 - np.mean(IQ3)
    print(f'IQ1长度{len(IQ1)}, IQ1长度{len(IQ2)}, IQ1长度{len(IQ3)}')

    # 插值（重采样）--插值这里也是有算法的

    # 快速傅里叶变换
    fft_1 = fft(IQ1)
    fft_2 = fft(IQ2)
    fft_3 = fft(IQ3)

    R12 = np.abs(ifft(IQ1 * IQ2))
    print(R12[:5])
    print(f'R12长度{len(R12)}')
    x = np.linspace(0, len(R12), len(R12))
    plt.scatter(x, R12)
    plt.show()

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

    # timedelay12 = calculateTimeDelay(I[:, 0], Q[:, 0], I[:, 1], Q[:, 1], subFs=1e8)
    # timedelay13 = calculateTimeDelay(I[:, 0], Q[:, 0], I[:, 2], Q[:, 2])
    # print(f'时间差1-2：{timedelay12}ns, 时间差1-3：{timedelay13}ns')


if __name__ == '__main__':
    main()
