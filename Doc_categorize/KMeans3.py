from scipy.cluster.vq import *
from scipy.misc import imresize

from pylab import *

from PIL import Image

#steps*steps像素聚类
def clusterpixels_square(infile, k, steps):

    im = array(Image.open(infile))

    #im.shape[0] 高 im.shape[1] 宽
    dx = im.shape[0] // steps
    dy = im.shape[1] // steps
    # 计算每个区域的颜色特征
    features = []
    for x in range(steps):
        for y in range(steps):
            R = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 0])
            G = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 1])
            B = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 2])
            features.append([R, G, B])
    features = array(features, 'f')     # 变为数组
    # 聚类， k是聚类数目
    centroids, variance = kmeans(features, k)
    code, distance = vq(features, centroids)

    # 用聚类标记创建图像
    codeim = code.reshape(steps, steps)
    codeim = imresize(codeim, im.shape[:2], 'nearest')
    return codeim

#stepsX*stepsY像素聚类
def clusterpixels_rectangular(infile, k, stepsX):

    im = array(Image.open(infile))

    stepsY = stepsX * im.shape[1] // im.shape[0]

    #im.shape[0] 高 im.shape[1] 宽
    dx = im.shape[0] // stepsX
    dy = im.shape[1] // stepsY
    # 计算每个区域的颜色特征
    features = []
    for x in range(stepsX):
        for y in range(stepsY):
            R = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 0])
            G = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 1])
            B = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 2])
            features.append([R, G, B])
    features = array(features, 'f')     # 变为数组
    # 聚类， k是聚类数目
    centroids, variance = kmeans(features, k)
    code, distance = vq(features, centroids)
      # 用聚类标记创建图像
    codeim = code.reshape(stepsX, stepsY)
    codeim = imresize(codeim, im.shape[:2], 'nearest')
    return codeim



#计算最优steps 为保证速度以及减少噪点 最大值为maxsteps 其值为最接近且小于maxsteps 的x边长的约数
def getfirststeps(img,maxsteps):

    msteps = img.shape[0]

    n = 2

    while(msteps>maxsteps):

        msteps =   img.shape[0]//n
        n = n + 1

    return msteps



#Test


#图像文件 路径
infile = 'stln.jpg'

im = array(Image.open(infile))

#参数
m_k = 3

m_maxsteps = 128

#显示原图empire.jpg
figure()

subplot(131)

title('source')

imshow(im)

# 用改良矩形块对图片的像素进行聚类
codeim= clusterpixels_rectangular(infile, m_k,getfirststeps(im,m_maxsteps))

subplot(132)

title('New steps = '+str(getfirststeps(im,m_maxsteps))+' K = '+str(m_k));

imshow(codeim)

#方形块对图片的像素进行聚类
st=100
codeim= clusterpixels_square(infile, 10, st)

subplot(133)

title('Old steps = 200,K = '+str(m_k))

imshow(codeim)

show()