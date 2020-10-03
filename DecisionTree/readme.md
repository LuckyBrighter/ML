1. 原理

决策树分类的思想很朴素，其核心是当前在众多的属性中，筛选出信息最为丰富对划分最为重要的属性，按照属性的不同取值对数据集划分，再递归对划分的每一个子数据集在剩下的属性中重新筛选最为重要的属性，继续划分子数据集。直到满足两个停止条件之一。

2. 实现

(1) 如何在众多属性中筛选出最为重要的属性？

通过信息增益，我们可以量化衡量一个属性对最终分类结果的贡献。
对于数据集D上，属性a的信息增益公式为![image](http://note.youdao.com/yws/public/resource/687a6027436fd1eb1ca20bed6f152d21/xmlnote/D0109F57DB434D079C98EF4A39665840/11828)
其中 Env(D^v^)为数据集D上属性为a的子数据集的信息熵，其公式为
![image](http://note.youdao.com/yws/public/resource/687a6027436fd1eb1ca20bed6f152d21/xmlnote/C4C77A67BBDE4C949764092B6F0D61F5/11842)

其中`$p_k$`为D中类别为k的样本占D中总样本的比例

计算一个数据集的香农熵的代码如下：
```
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
```
筛选出信息增益最大的属性代码：
```
def chooseBestFeatureToSplit(self,dataSet):
		num_samples = len(dataSet)
		featrue_num = len(dataSet[0])-1 # 减去最后一列的分类label
		max_gain = 0.0
		best_fea = -1
		base_ent = self.countShannonEnt(dataSet,num_samples,-1)
		#print("featrue_num={}".format(featrue_num))
		for axis in range(featrue_num):#featrue_num个特征
			#每个特征的属性
			featrue_value = [sample[axis] for sample in dataSet]
			featrue_value = set(featrue_value)
			ent = 0.0
			for featrue in featrue_value:
				subData = self.splitData(dataSet,axis,featrue)
				num_sub = len(subData)
				shannon = self.countShannonEnt(subData,num_sub,-1)
				#print("featrue_value[{}] = {} , shannonEnt = {}".format(axis,featrue,shannon))
				ent += float(num_sub)/(num_samples) * shannon
			gain = base_ent - ent
			#print("featrue {} shannonEnt = {}".format(axis,ent))
			#print("featrue {} gain = {}".format(axis,gain))
			if gain > max_gain:
				max_gain = gain
				best_fea = axis
		return best_fea
```

(2) 对于某一属性缺失值，如何处理？

在数据集D中，将属性a值为空的样本从D中删除得到数据集为`$D_r$`,对`$D_r$`计算属性a的信息增益。

按照上述方法，筛选出信息增益最大的属性，记为C，根据C的不同取值，分割数据集，将属性C缺失的样本都划入分割的子数据集中。

例如C的取值为{0,1,2},把数据集划分为`$D_1$`,`$D_2$`,`$D_3$`，那么把C缺失的样本都加入数据集`$D_1$`,`$D_2$`,`$D_3$`中。


(3) 递归建树
```
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
	
```


### 参考：
1. 《机器学习实战》
2. [西瓜数据集](https://blog.csdn.net/Leafage_M/article/details/79560791)
3. [绘制字典形式的决策树](https://blog.csdn.net/sinat_29957455/article/details/76553987)

