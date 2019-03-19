#coding=utf-8
from numpy import *
import random;
import struct;
import numpy;



def Get_Data_CLass1():
    f = open('data_class1.txt', 'rb');
    f.seek(0,0);

    res=numpy.zeros([1024,8]);

    for i in range(0,1024):
        for j in range (0,8):
            bytes=f.read(4);
            fvalue,=struct.unpack("f",bytes);
            res[i][j]=fvalue;
    return res;
def Get_Data_CLass2():
    f = open('data_class2.txt', 'rb');
    f.seek(0,0);

    res=numpy.zeros([1024,8]);

    for i in range(0,1024):
        for j in range (0,8):
            bytes=f.read(4);
            fvalue,=struct.unpack("f",bytes);
            res[i][j]=fvalue;
    return res;


'''pca函数有两个参数，其中dataMat是已经转换成矩阵matrix形式的数据集，列表示特征；
k表示要降到的维度数'''
def pca(dataMat,k):
    # 对每一列求平均值
    meanVals=mean(dataMat,axis=0)   #shape:1*8
    print('1',meanVals.shape)
    meanRemoved=dataMat-meanVals    #shape:1024*8
    print('2',meanRemoved.shape)
    # cov()计算协方差矩阵
    covMat=cov(meanRemoved,rowvar=0)    #shape:8*8
    print('3',covMat.shape)
    # 用numpy模块linalg中的eig()方法求特征值和特征向量
    eigVals,eigVects=linalg.eig(mat(covMat))    #特征向量eigVects的shape:8*8
    # 对特征值eigVals从小到大排序
    eigValInd=argsort(eigVals)
    print('4',eigVects.shape)
    # 从排好序的特征值，从后往前取k个，这样就实现了特征值的从大到小排列
    eigValInd=eigValInd[:-(k+1):-1]

    # 返回排序后特征值对应的特征向量redEigVects（主成分）
    redEigVects=eigVects[:,eigValInd]   #shape:8*k
    print('6',redEigVects.shape)
    # 将原始数据投影到主成分上得到新的低维数据lowDDataMat
    lowDDataMat=meanRemoved*redEigVects     #shape:1024*k
    print('7',lowDDataMat.shape)
    # 得到重构数据reconMat
    #reconMat=(lowDDataMat*redEigVects.T)+meanVals      #shape:1024*8
    return lowDDataMat



dataMat1=Get_Data_CLass1()
dataMat2=Get_Data_CLass2()
lowDDataMat1=pca(dataMat1,4)
lowDDataMat2=pca(dataMat2,6)
print(lowDDataMat1)
print(lowDDataMat2)

