#!C:\pythonCode
# -*- coding: utf-8 -*-
# @Time : 2022/10/15 14:43
# @Author : zmz
# @File : mat2npy.py
# @Software: PyCharm
import numpy as np
from scipy import io

mat = io.loadmat('point1.mat')
print(type(mat))
matt = dict([(key, mat[key]) for key in ['wrd','zmz','zy']])
print(matt)
# 如果报错:Please use HDF reader for matlab v7.3 files
# 改为下一种方式读取
# import h5py
# mat = h5py.File('point1.mat')

# mat文件里可能有多个cell，各对应着一个dataset

# 可以用keys方法查看cell的名字, 现在要用list(mat.keys()),
# 另外，读取要用data = mat.get('名字'), 然后可以再用Numpy转为array
# print(mat.keys())
# 可以用values方法查看各个cell的信息
# print(mat.values())

# 可以用shape查看维度信息
# print(mat['zmz'])
# 注意，这里看到的shape信息与你在matlab打开的不同
# 这里的矩阵是matlab打开时矩阵的转置
# 所以，我们需要将它转置回来
# mat_t = np.transpose(mat['point1'])
# mat_t 是numpy.ndarray格式

# 再将其存为npy格式文件
np.save('point1.npy', matt)
#