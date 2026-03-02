from DataAnalysis import *
import numpy as np
import matplotlib.pyplot as plt

class PicDraw(Data_Analysis):
    def __init__(self,filepath):
        super().__init__(filepath)
        plt.rcParams['font.size'] = 20
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['ytick.major.width'] = 1.0
        plt.rcParams['xtick.major.width'] = 1.0
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.figsize'] = (7,7)
        plt.rcParams['figure.subplot.bottom'] = 0.15
        plt.rcParams['figure.subplot.left'] = 0.15
        plt.rcParams['figure.subplot.right'] = 0.90
        plt.rcParams['figure.subplot.top'] = 0.90

#更改全局设置
    def global_param(self,group,number):
        plt.rcParams[group] = number

#直方图绘制
    def hist_draw(self, data, ymax, bin_width,  ymin = 0,
                  xticknumber = 5,yticknumber= 5,xspace = 1000,
                  guassfitting=False,
                  histcolor = 'skyblue'):
    #数据筛选
        x = data['data']

    #设置bin值
        bin_edges = np.arange(data['mindata'], data['maxdata'] + bin_width, bin_width)

        #高斯曲线
        if guassfitting == True:
            #拟合
            fitting_params = self.fitting_params(x,bin_edges)

            #峰值
            peak_number = "{:.1f}".format(fitting_params[2]) + ' ' + data['unit']
            plt.text(x=(fitting_params[2]-data['mindata'])*0.6,
                     y=fitting_params[1]+((ymax-fitting_params[1])*0.2),
                     s=peak_number,color='black')

            #作图
            x_fitting = np.linspace(data['mindata'],data['maxdata'],xspace)
            y_fitting = self.gaussamp_function(fitting_params, x_fitting)
            plt.plot(x_fitting,y_fitting,lw=2.0,color = 'black')

        #绘制直方图
        plt.hist(x,bins=bin_edges, color=histcolor)

        #设置坐标轴
        plt.xlim(data['mindata'], data['maxdata'])
        plt.xticks(np.arange(data['mindata'], data['maxdata']+bin_width, (data['maxdata']-data['mindata'])/(xticknumber-1)))
        plt.xlabel(data['name']+' '+'/'+' '+data['unit'])
        plt.ylim(ymin, ymax)
        plt.yticks(np.arange(ymin, ymax+1, (ymax-ymin)/(yticknumber-1)))
        plt.ylabel('Counts')

        plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        # plt.show()

    #设置坐标轴
    def xaxis_title(self,xaxistitle,yaxistitle):
        plt.ylabel(yaxistitle)
        plt.xlabel(xaxistitle)

    #设置注释
    def pic_note(self,x,y,content):
        plt.text(x=x,y=y,s=content,color='black')

    # 保存图片并清除
    def pic_save(self,filename):
        #plt.show()
        plt.savefig(filename+'.tiff')
        plt.clf()

    #清除当前作图
    def cls_pic(self):
        plt.clf()



if __name__ == '__main__':
    #读取数据
    Result = PicDraw('E:/Doctorate/Subject/纳米电极电化学响应单分子拉曼测量/实验数据处理/20251010/mV600.xlsx')
    #print(Result.row_data(2))

    #计算时间
    data1 = Result.coldata_chosen('Time',datamax=4)
    Result.hist_draw(data1,20,0.1,guassfitting=True,
                     histcolor=(9/255, 147/255, 150/255, 0.5),
                     xticknumber=5,yticknumber= 5)
    Result.pic_save(data1['name'])

    #计算电流
    data2 = Result.coldata_chosen('DeltaI', datamax=600)
    Result.hist_draw(data2, 40, 10, guassfitting=True,
                     histcolor=(235/255, 215/255, 165/255, 1),
                     xticknumber=5,yticknumber= 5)
    Result.pic_save(data2['name'])

    #计算电荷
    data3 = Result.coldata_chosen('Charge', datamax=100)
    Result.hist_draw(data3, 40, 5, guassfitting=True,
                     histcolor=(78/255, 166/255, 96/255, 0.5),
                     xticknumber=6,yticknumber=5)
    Result.pic_save(data3['name'])