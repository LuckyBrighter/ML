# 解决数据丢失的数据集的分类问题
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import math
class DecisionTree(object):
	"""docstring for DecisionTree"""
	def countShanonEnt(self, dataSet):
		classes = dict()
		for data in dataSet:
			if data[-1] not in classes:
				classes[data[-1]] = 0
			classes[data[-1]] += 1
		# 计算类别数所占比例以及信息熵
		ent = 0
		num_samples = len(dataSet)
		for key in classes.keys():
			prob =  float(classes[key])/num_samples
			ent -= prob*math.log(prob,2)
		return ent

	# 根据属性feature[feature_index]划分数据集 同时处理缺失值
	def splitData(self,dataSet,feature_index,feature_value):
		subDataSet = []
		for data in dataSet:
			if data[feature_index] == feature_value:
				subData = data[:feature_index]
				subData.extend(data[feature_index+1:]) #extend可以将一个列表扩充到另一个列表的末尾	
				#print(subData)
				subDataSet.append(subData)
		#print(subDataSet)
		return subDataSet

	def chooseBestFeature(self,dataSet):
		#计算属性各个属性的信息增益，取最大者作为当前的划分属性
		best_feature_index = -1
		max_gain = 0
		num_samples = len(dataSet)
		if num_samples == 0:
			print('error:the size of dataSet is zero')
			return 
		num_featues = len(dataSet[0]) - 1 
		base_ent = self.countShanonEnt(dataSet)
		for index in range(num_featues):
			missing_dataSet = self.getMissDataSet(dataSet,index)
			featrue_value = []
			for data in dataSet:
				if data[index] != '-':
					featrue_value.append(data[index])
			#featrue_value = [data[index] for data in dataSet]
			featrue_value = set(featrue_value)
			ent = 0.0
			not_missing_num = num_samples - len(missing_dataSet)
			for value in featrue_value:
				subDataSet = self.splitData(dataSet,index,value)
				prob = float(len(subDataSet))/not_missing_num
				shannonEnt = self.countShanonEnt(subDataSet)
				ent += prob*shannonEnt
			gain = base_ent - ent
			if gain > max_gain:
				max_gain = gain
				best_feature_index = index
		return best_feature_index
	
	def getMissDataSet(self,dataSet,feature_index):
		missDataSet = []
		for data in dataSet:
			if data[feature_index] == '-':
				subData = data[:feature_index]
				subData.extend(data[feature_index+1:])
				missDataSet.append(subData)
		return missDataSet

	def createTree(self,dataSet,features):
		#停止条件：
		#01 当前类数据集的类标签全都相同
		#02 所有特征属性全部筛选完，选择标签类别最多的标签返回
		#print(features)
		class_label = [data[-1] for data in dataSet]
		#print(class_label)
		if class_label.count(class_label[0]) == len(dataSet):
			return class_label[0]
		if len(dataSet[0]) == 1:
			return self.majority(dataSet) 
		#若未停止，则逐层拆分数据集
		best_feature_index = self.chooseBestFeature(dataSet)
		best_feature_name = features[best_feature_index]
		#print("best_feature_index={},best_feature_name={}".format(best_feature_index,best_feature_name))
		feature_values = []#[data[best_feature_index] for data in dataSet]
		for data in dataSet:
			if data[best_feature_index] != '-':
				feature_values.append(data[best_feature_index])
		feature_values = set(feature_values)
		#print("feature_values{}".format(feature_values))
		myTree = {(best_feature_name):{}}
		# 扩展子节点，边为属性值
		del features[best_feature_index]
		missDataSet = self.getMissDataSet(dataSet,best_feature_index)
		for value in feature_values:
			subData = self.splitData(dataSet,best_feature_index,value)
			#print(subData)
			subData.extend(missDataSet)
			subFeatures = features[:]
			#print(subData)
			myTree[(best_feature_name)][value] = self.createTree(subData,subFeatures)
		return myTree
	
	def majority(self,dataSet):
		class_count = dict()
		for data in dataSet:
			if data[-1] not in class_count:
				class_count[data[-1]] = 0
			class_count[data[-1]] += 1
		majority_class = dataSet[0][-1]
		max_count = class_count[majority_class]
		for key,count in class_count:
			if count >= max_vote:
				max_count = count
				majority_class = key
		return majority_class

	

	def storeTree(self,inputTree,filename):
		import pickle
		fw = open(filename,'wb+')
		pickle.dump(inputTree,fw)
		fw.close()
