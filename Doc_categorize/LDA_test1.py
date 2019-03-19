#coding=utf-8
from numpy import *
import random;
import struct;
import numpy;
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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


def lda(c1,c2,k):
    #c1 第一类样本，每行是一个样本
    #c2 第二类样本，每行是一个样本
    #计算各类样本的均值和所有样本均值
    m1=mat(mean(c1,axis=0))#第一类样本均值
    m2=mat(mean(c2,axis=0))#第二类样本均值
    c=vstack((c1,c2))#所有样本
    m=mean(c,axis=0)#所有样本的均值
    #计算类内离散度矩阵Sw
    n1=c1.shape[0]#第一类样本数
    n2=c2.shape[0]#第二类样本数
    d = c1.shape[1]
    Sb = numpy.mat(numpy.zeros((d, d)))
    S1 = ((c1- m1).T).dot(c1-m1)
    S2 = ((c2 - m2).T).dot(c2 - m2)
    Sw = S1 + S2                                    #Sw的shape:8*8
    #计算类间散度矩阵Sb
    Sb=((m-m1).T).dot(m-m1)+((m-m2).T).dot(m-m2)    #Sb的shape:8*8
    S=linalg.inv(Sw).dot(Sb)                        #S的shape:8*8
    #计算S的特征值和特征向量
    eigVals, eigVects=linalg.eig(S)
    # 对特征值eigVals从小到大排序
    eigValInd = argsort(eigVals)
    # 从排好序的特征值，从后往前取k个
    eigValInd = eigValInd[:-(k + 1):-1]
    redEigVects = eigVects[:, eigValInd]            # shape:8*k
    # 将原始数据投影得到降维后的数据lowDDataMat
    lowDDataMat1 = c1 * redEigVects                 # shape:1024*k
    lowDDataMat2 = c2 * redEigVects                 # shape:1024*k
    return lowDDataMat1,lowDDataMat2

if __name__=='__main__':
    dataMat1=Get_Data_CLass1()
    dataMat2=Get_Data_CLass2()
    lowDDataMat1,lowDDataMat2=lda(dataMat1,dataMat2,4)
    print(lowDDataMat1)
    print(lowDDataMat2)

