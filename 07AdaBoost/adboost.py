import numpy as np
import math
def stumpClassify(dataMatrix, dim, threshVal, threshIneq):
    retArray = np.ones((dataMatrix.shape[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dim] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dim] > threshVal] = -1.0
    return retArray

# 弱学习器
def bulidStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m,n = dataMatrix.shape
    numSteps = 10.0
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m,1)))
    minError = np.inf
    for i in range(n):
        rangeMin = np.min(dataMatrix[:,i])
        rangeMax = np.max(dataMatrix[:,i])
        # 步长
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(0, int(numSteps)+1):
            for inequal in ['lt','gt']: #两种方法
                threshVal = (rangeMin + float(j) * stepSize) # 以stepSize为步长获得在[minRange,maxRange]中的10个数
                # 以threshVal为阈值，小于threshVal的预测值为0,大等于threshVal的预测值为1
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = np.ones((m,1))
                #计算错误率
                errArr[predictedVals == labelMat] = 0
                #错误率权重 D为m×1的矩阵
                #print("D.T = {}".format(D.T))
                #print("errArr = {}".format(errArr))
                #D与errArr向量作内积
                weightedError = D.T * errArr
                print("split : dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f"%(i,threshVal,inequal,weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i # 当前分类错误率最小对应的分类特征
                    bestStump['thresh'] = threshVal # 当前分类错误最小对应的阈值
                    bestStump['inequal'] = inequal # 当前分类错误率阈值处理方法
    return bestStump, minError, bestClassEst

def adaBoostTrainDS(dataArr, classLabels, numInt = 40):
    weakClassArr = []
    m = dataArr.shape[0]
    D = np.zeros((m,1)) #D在迭代过程中需要优化的
    aggClassEnt = np.mat(np.zeros((m,1)))
    for i in range(numInt):
        bestStump, error, classEst = bulidStump(dataArr,classLabels,D)
        print("D:",D.T)
        # 为什么是max(error,1e-16)
        alpha = float(0.5 * math.log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        # 弱学习器添加
        weakClassArr.append(bestStump)
        # 预测分类
        print("classEst: ",classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, math.exp(expon))
        D = D/D.sum()
        aggClassEnt += alpha * classEst
        print("aggClassEst : ",aggClassEnt.T)
        # 累加的当前弱分类器的错误率
        aggErrors = np.multiply(math.sign(aggClassEnt)!= np.mat(classLabels).T, np.ones((m,1)))
        errorRate = aggErrors.sum()/m
        print("total error: ", errorRate)
        if errorRate == 0.0:
            break # python浮点数的比较？
    return weakClassArr



