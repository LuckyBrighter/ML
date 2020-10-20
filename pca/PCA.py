import numpy as np

def pca(dataMat, topNfeat):
    #对输入数据进行标准化,为什么不除以样本方差呢?
    meanVals = np.mean(dataMat, axis = 0)
    meanRemoved = dataMat - meanVals
    #求解标准化数据的协方差矩阵,rowvar=False表示列数据为同一个变量的观测值
    covMat = np.cov(meanRemoved,rowvar = False)
    #计算协方差矩阵的特征值,特征向量.其中协方差的特征向量表示的是使得y=a*x线性变换,var(y)最大化的解.
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    #从小到大排列特征值,得到对应下标序列
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[-(topNfeat+1):-1]

    A = eigVects[:,eigValInd]
    lowDataMat = np.matmul(meanRemoved,A)
    reconMat = (lowDataMat * A.T) + meanVals
    #return lowDataMat,reconMat
    return np.sort(eigVals)
#调整主成分变量的数目
def adjustK(dataMat,maxNum = 20):
    X = [ i+1 for i in range(maxNum)]
    Y = []
    for i in range(maxNum):
        eigVals = pca(dataMat,i+1)
        sum = np.sum(eigVals)
        principle_sum = np.sum(eigVals[-(i+1):])
        Y.append(float(principle_sum/sum))
    print(Y)
    return X,Y
import matplotlib.pyplot as plt
#调整不同的主成分数目,看其方差的百分比变化
def view(x,y):
    from matplotlib import font_manager
    plt.xlabel('principle component num')
    plt.ylabel('contribution rate')
    plt.plot(x, y, color="r", linestyle="--", marker="*", linewidth=1.0)
    plt.show()