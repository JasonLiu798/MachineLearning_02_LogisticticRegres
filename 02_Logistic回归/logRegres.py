#coding:utf-8
from numpy import *
import matplotlib.pyplot as plt

class func:
    def loadDataSet(self):
        dataMat = []
        labelMat = []
        fr = open('testSet.txt')
        for line in fr.readlines():
            # 矩阵的格式【标签，X1，X2】
            lineArr = line.strip().split()

            #插入X1，X2，以及初始化的回归系数（权重），全部设为1.0
            dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])

            #插入标签,
            #必须转换成int类型，因为只有两类
            labelMat.append(int(lineArr[2]))


        return dataMat,labelMat

    # 使用sigmoid函数进行分类
    def sigmoid(self,inX):
        return 1.0/(1+exp(-inX))

    #使用梯度上升算法进行优化，找到最佳优化系数
    #dataMatIn是训练集，三维矩阵，（-0.017612	14.053064	0）
    #classLabels是类别标签，数值型行向量，需要转置为列向量,取值是0,1
    def gradAscent(self,dataMatIn,classLabels):
        dataMatrix = mat(dataMatIn) #转换为Numpy矩阵数据类型
        labelMat = mat(classLabels).transpose() #转置矩阵,便于加权相乘
        m,n = shape(dataMatrix) #得到矩阵的行列数，便于设定权重向量的维度

        alpha = 0.001
        maxCycles = 500
        weights = ones((n,1))  #返回一个被1填满的 n * 1 矩阵，作为初始化权重
        for k in range(maxCycles):
            h = self.sigmoid(dataMatrix*weights)
            erro = (labelMat - h)   #labelMat是使用当前回归系数得到的的标签，labelMat是训练集的标签，取值是0,1
            weights = weights + alpha * dataMatrix.transpose( ) * erro  #根据使用当前权重计算的值与初始值的误差，更改weight，
                                                                        #按照误差的方向调整回归系数
        return weights

    def plotBestFit(self,wei):
        weights = wei.getA()
        dataMat,labelMat = self.loadDataSet()
        dataArr = array(dataMat)
        n = shape(dataMat)[0]
        xcord1 = []
        ycord1 = []
        xcord2 = []
        ycord2 = []

        #根据标签不同分为两种颜色画图，易于分辨
        for i in range(n):
            if int(labelMat[i]) == 1:
                xcord1.append(dataArr[i,1])
                ycord1.append(dataArr[i,2])
            else :
                xcord2.append(dataArr[i, 1])
                ycord2.append(dataArr[i, 2])

        fig = plt.figure()
        ax = fig.add_subplot(111) #将画布分为一行一列的画布，并从第一块开始画
        ax.scatter(xcord1,ycord1,s=30,c='red',marker = 's') #scatter散开
        ax.scatter(xcord2,ycord2,s=30,c='green')
        x = arange(-3.0,3.0,1)
        y = (-weights[0]-weights[1]*x)/weights[2]
        ax.plot(x,y)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()




if __name__=="__main__":
    logRegres = func()
    dataArr,labelMat = logRegres.loadDataSet()
    weights = logRegres.gradAscent(dataArr,labelMat)
    logRegres.plotBestFit(weights)











