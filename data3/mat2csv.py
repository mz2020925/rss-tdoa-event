#!C:\pythonCode
# -*- coding: utf-8 -*-
# @Time : 2022/10/15 17:50
# @Author : zmz
# @File : mat2csv.py
# @Software: PyCharm
import pandas as pd
from scipy import io
features_struct = io.loadmat('point1.mat')
features = features_struct['wrd']
# print(features_struct)
dfdata = pd.DataFrame(features)
print(dfdata)
# datapath1 = 'point1.csv'
# dfdata.to_csv(datapath1, index=False)

