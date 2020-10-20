import numpy as np

def loadDataSet(filename,delim = '\t'):
    file = open(filename)
    # line.strip() Remove spaces at the beginning and at the end of the string:
    dataArr = []
    for line in file.readlines():
        temp_list = line.strip().split()
        temp_list = [ e if e != 'NaN' else e for e in temp_list]
        dataArr.append(temp_list)
    return np.array(dataArr)

# python 的文件处理

#将每一列的均值替换缺失值NaN
def replaceNanWithMean(filenmae):
    dataArr = loadDataSet(filenmae)
    # 对每一列的数据取均值
    sample_num = dataArr.shape[0]
    fea_num = dataArr.shape[1]
    for j in range(fea_num):
        sum = 0
        num = 0
        nan_pos = []
        for i in range(sample_num):
            if dataArr[i][j] != 'NaN':
                sum += float(dataArr[i][j])
                num += 1
            else:
                nan_pos.append(i)
        mean = float(sum)/num
        #print(nan_pos)
        for pos in nan_pos:
            dataArr[pos][j] = mean
    #print(dataMat)
    return np.matrix(dataArr.astype(float))
