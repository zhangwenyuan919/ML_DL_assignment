#coding=utf-8
from numpy import *
import random;
import struct;
import numpy;
from sklearn.cross_decomposition import CCA



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



def cca(x,y):
    # 对每一列求平均值
    mean1 = mean(x,axis=0)                  #shape:1*8
    mean2 = mean(y, axis=0)                 # shape:1*8
    #对原始数据进行标准化，使其均值为0
    meanX = x - mean1                       #shape:1024*8
    meanY = y - mean2                       #shape:1024*8
    #求Sxx,Syy,Sxy,Syx
    Sxx = cov(meanX, rowvar=0)
    Syy = cov(meanY, rowvar=0)
    Sxy = cov(meanX, meanY, rowvar=0)
    Syx = cov(meanY, meanX, rowvar=0)
    #求Sxx的逆，Syy的逆
    Sxx_1=linalg.inv(Sxx);
    Syy_1=linalg.inv(Syy);
    mat1=Sxx_1.dot(Sxy).dot(Syy_1).dot(Syx)
    mat2=Syy_1.dot(Syx).dot(Sxx_1).dot(Sxy)
    #对mat1,mat2进行奇异值分解
    w=linalg.svd(mat1)
    v=linalg.svd(mat2)
    return w,v




