

#decisionTree = DecisionTree.DecisionTree()
#myTree = decisionTree.grabTree('decisionTreeStorage.txt')
#print(myTree)
#数据集参考博客 https://blog.csdn.net/Leafage_M/article/details/79560791

def createDataSet():
    """
    创建测试的数据集
    :return:
    """
    dataSet = [
        # 1
        ['-', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 2
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '-', '好瓜'],
        # 3
        ['乌黑', '蜷缩', '-', '清晰', '凹陷', '硬滑', '好瓜'],
        # 4
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        # 5
        ['-', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 6
        ['青绿', '稍蜷', '浊响', '清晰', '-', '软粘', '好瓜'],
        # 7
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
        # 8
        ['乌黑', '稍蜷', '浊响', '-', '稍凹', '硬滑', '好瓜'],

        # ----------------------------------------------------
        # 9
        ['乌黑', '-', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
        # 10
        ['青绿', '硬挺', '清脆', '-', '平坦', '软粘', '坏瓜'],
        # 11
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '-', '坏瓜'],
        # 12
        ['浅白', '蜷缩', '-', '模糊', '平坦', '软粘', '坏瓜'],
        # 13
        ['-', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
        # 14
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
        # 15
        ['乌黑', '稍蜷', '浊响', '清晰', '-', '软粘', '坏瓜'],
        # 16
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
        # 17
        ['青绿', '-', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
    ]
    return dataSet

import DecisionTree

def createDataSet1():
	dataSet = [
			[1,1,'yes'],
			[1,1,'yes'],
			[1,0,'no'],
			[0,1,'no'],
			[0,1,'no']
	]
	labels = ['no surfacing','flippers']
	return dataSet,labels

def grabTree(filename):
	import pickle
	fr = open(filename,'rb+')
	return pickle.load(fr,encoding = 'utf-8')

def classify(inputTree, featLabels, testVec,features_index_dict):
	rootStr = list(inputTree.keys())[0]
	subTrees = inputTree[rootStr]
	# 找到标签rootStr在测试集中的哪一个位置
	#print(f)
	featIndex = features_index_dict[rootStr]
	#print("featIndex={}".format(featIndex))
	class_label = "西瓜"
	for key in subTrees.keys():
		#print("featname={},testValue={},child={}".format(rootStr[0],testVec[featIndex],key))
		if testVec[featIndex] == key:
			if type(subTrees[key]).__name__ == 'dict':
				# 非叶子节点
				classLabel = classify(subTrees[key],featLabels,testVec,features_index_dict)
			else:				
				classLabel = subTrees[key]
	return classLabel


def main():

	features = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感']
	features_index_dict = {'色泽':0, '根蒂':1, '敲击':2, '纹理':3, '脐部':4, '触感':5}
	filename = "decisionTree.txt"
	myTree = None
	import os
	if not os.path.exists(filename):
		dataSet = createDataSet()
		decisionTree = DecisionTree.DecisionTree()
		myTree = decisionTree.createTree(dataSet,features)
		print("not exists")
		decisionTree.storeTree(myTree,filename)
	else:
		myTree = grabTree(filename)

	import treePlot
	treePlot.createPlot(myTree)
	testVec = ['-', '蜷缩', '浊响', '清晰', '凹陷', '硬滑']
	result = classify(myTree,features,testVec,features_index_dict)
	print(result)

main()