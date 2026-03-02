import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import alpha
import csv


class Data_Analysis:
    def __init__(self,filepath):
        self.filepath = filepath
        self.data = pd.read_excel(self.filepath)
        self.colname = self.data.columns[:]

#选择行数据
    def row_data(self,rownumber):
        return self.data.iloc[rownumber-1]

#选择列数据以及单位
    def coldata_chosen(self,colname,datamax=0,datamin=0):
        self.anadata = {}
        self.anadata['name'] = colname
        self.anadata['data'] = self.data[colname]
        if colname in ['Current','Baseline','DeltaI']:
            self.anadata['unit'] = 'pA'
        elif colname in ['Time','InitialTime']:
            self.anadata['unit'] = 'ms'
        elif colname in ['Charge']:
            self.anadata['data'] = self.data[colname]*1000
            self.anadata['unit'] = 'fC'
        elif colname in ['I/I0']:
            self.anadata['unit'] = ''

        #筛选数据
        if datamax < datamin:
            print('data error')
        elif datamax > datamin:
            self.anadata['data'] = self.anadata['data'][(self.anadata['data'] <= datamax) & (self.anadata['data'] >= datamin)]
            self.anadata['maxdata'] = datamax
            self.anadata['mindata'] = datamin
        elif datamax == 0 & datamin == 0:
            self.anadata['maxdata'] = max(self.anadata['data'])
            self.anadata['mindata'] = min(self.anadata['data'])

        return self.anadata

    #选择数据循环

#L-M拟合高斯函数
    def gaussamp_function(self,params,x):
        y0,Amp,mu,sigma = params
        return y0 + Amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    def residuals(self, params, x, y):
        return self.gaussamp_function(params,x) - y

    def fitting_params(self,data,bins,initial_params=[0,0,0,0]):
        #直方图bin值
        counts, bin_edges, _ = plt.hist(data, bins=bins)
        y = counts
        x = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.clf()
        #print(x,y)

        #初始参数猜测
        initial_params[1] = np.max(y)
        initial_params[2] = x[np.argmax(y)]
        initial_params[3] = np.std(y)
        #print(initial_params)

        #拟合参数
        result = least_squares(self.residuals, initial_params, args=(x, y), method='trf')
        return result.x


if __name__ == '__main__':
    print('yes')