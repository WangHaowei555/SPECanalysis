from venv import CORE_VENV_DEPS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import alpha


class CV_analysis:
    def __init__(self, filepath):
        self.filepath = filepath
        self.potentialrange = {}
        self.currentrange = {}

        #读取设置
        self.setup = pd.read_csv(self.filepath, header=None, skiprows=7, nrows=9, delimiter='=')
        self.init_potential = np.float64(self.setup[1][0])
        self.high_potential = np.float64(self.setup[1][1])
        self.low_potential = np.float64(self.setup[1][2])
        self.segment = int(self.setup[1][5])
        self.sample_interval = np.float64(self.setup[1][6])
        self.sensitivity = np.float64(self.setup[1][8])

        #读取数据
        file = pd.read_csv(self.filepath, header=None)
        skiprows = 0
        for row in file[0]:
            if row != 'Potential/V':
                skiprows = skiprows + 1
            else:
                break
        self.data = pd.read_csv(self.filepath, header=skiprows)

        #电流电位轴
        self.potentialrange['data'] = self.data['Potential/V']
        self.potentialrange['unit'] = 'V'
        if self.sensitivity < 1e-9 and self.sensitivity >= 1e-12:
            self.currentrange['data'] = self.data[' Current/A'] * 1e12
            self.currentrange['unit'] = 'pA'
        elif self.sensitivity < 1e-6 and self.sensitivity >= 1e-9:
            self.currentrange['data'] = self.data[' Current/A'] * 1e9
            self.currentrange['unit'] = 'nA'
        elif self.sensitivity < 1e-3 and self.sensitivity >= 1e-6:
            self.currentrange['data'] = self.data[' Current/A'] * 1e6
            self.currentrange['unit'] = '\u03bc' + 'A'

        #内置参数
        self.__datapoints__ = int((self.high_potential - self.low_potential) / self.sample_interval)
        plt.rcParams['font.size'] = 20
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['ytick.major.width'] = 1.0
        plt.rcParams['xtick.major.width'] = 1.0
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.figsize'] = (7, 4)
        plt.rcParams['figure.subplot.bottom'] = 0.2
        plt.rcParams['figure.subplot.left'] = 0.2
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95

    #极限电流
    def limit_current(self, reactiontype, segment):
        if reactiontype == 'oxidation':
            LimitedCurrent = max(
                self.currentrange['data'][self.__datapoints__ * (segment - 1):self.__datapoints__ * (segment + 1) - 1])
        elif reactiontype == 'reduction':
            LimitedCurrent = min(
                self.currentrange['data'][self.__datapoints__ * (segment - 1):self.__datapoints__ * (segment + 1) - 1])

        return LimitedCurrent

    #CV绘图
    def CV_draw(self, segment, reactiontype=None):
        if segment > self.segment - 1 or segment <= 0:
            print('out of range!')
        else:
            #设置坐标轴
            x = self.potentialrange['data'][self.__datapoints__ * (segment - 1):self.__datapoints__ * (segment + 1) - 1]
            y = self.currentrange['data'][self.__datapoints__ * (segment - 1):self.__datapoints__ * (segment + 1) - 1]

            #绘图
            plt.plot(x, y, lw=2.0, color='black')

            if reactiontype == 'oxidation' or reactiontype == 'reduction':
                plt.text(x=0.3,
                         y=0.2,
                         s="{:.1f}".format(self.limit_current(reactiontype,segment)) + ' ' + self.currentrange['unit'],
                         color='black')

            plt.xlim(self.low_potential - 0.05, self.high_potential + 0.05)
            plt.xlabel('Potential' + ' ' + '/' + ' ' + 'V')
            plt.ylabel('Current' + ' ' + '/' + ' ' + self.currentrange['unit'])

            plt.show()

    def CV_drawall(self, reactiontype=None):
        for i in range(1, self.segment,2):
            self.CV_draw(i, reactiontype)


if __name__ == '__main__':
    CV1 = CV_analysis('E:/Doctorate/Subject/纳米电极电化学响应单分子拉曼测量/电极制备/Nanoelectrode/CV测试/20260209/AuNE20260208_5_1.csv')
    CV1.CV_drawall('oxidation')
    # CV1.CV_draw(7,'oxidation')