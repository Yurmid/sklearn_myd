#!/usr/bin/env python
# coding: utf-8

# In[3]:


#决策树是往信息增益最大的方向拟合 计算全部特征的不纯度指标  --> 选取关键点为分支点  -->>  不纯度指标最优


# In[4]:


# 分类树流程
# clf = tree.DecisionTreeClassifier() 实例化对象
# clf = clf.fit(x_train,y_train) 训练集
# result = clf.score(x_test,y_test) 测试集


# In[138]:


from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np


# In[17]:


wine = load_wine()


# In[27]:


wine.data


# In[25]:


wine.target


# In[20]:


wine.data.shape


# In[80]:


import pandas as pd
pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis=1)


# In[81]:


wine.feature_names


# In[82]:


wine.target_names


# In[83]:


Xtrain,Xtest,Ytrain,Ytest = train_test_split(wine.data,wine.target,test_size=0.3)


# In[84]:


#Xtrain.shape


# In[85]:


#Ytrain.shape


# In[128]:


#如何确定最优的剪枝参数？
import matplotlib.pyplot as plt
test = []
for i in range(50):
    clf = tree.DecisionTreeClassifier(criterion='entropy'
                                 ,random_state=i  #训练次数
                                 ,splitter='random' #所有特征
                                 ,max_depth= 3 #剪枝深度
                                )
    clf = clf.fit(Xtrain,Ytrain)
    score = clf.score(Xtest,Ytest)
    test.append(score)
plt.plot(range(0,50),test,color='red',label="random_state")
plt.legend()
plt.show()


# In[129]:



clf = tree.DecisionTreeClassifier(criterion='entropy'
                                 ,random_state=30  #训练次数
                                 ,splitter='random' #所有特征
                                 ,max_depth= 3 #剪枝深度
                            #     ,min_samples_leaf= 10  #划定每个分支节点下面的子节点至少包含的训练样本 不然就都砍掉
                            #     ,min_samples_split= 10  #划定每个分支节点至少包含的训练样本 不然就不分
                            #     ,max_features=  #限制特征个数，在高维度方式下降低过拟合，建议不用，用pca
                            #     ,min_impurity_decrease= #限制特征增益最小值，太小的没用，就不算了 
                            #     ,class_weight=   #完成样本平衡的参数   例子：银行违约本来就很少，就算全部预测否，正确率也很高
                            #因此要给少量标签更大的权重，让模型偏向少数类，默认为None，自动修正，调的话输比例
                            #     ,min_weight_fraction_leaf=  #如果上面设置了参数，则要使用基于权重的参数修剪
                                 )
clf = clf.fit(Xtrain,Ytrain)
score = clf.score(Xtest,Ytest)
score


# In[130]:


#考虑对训练集的拟合程度
score_train = clf.score(Xtrain,Ytrain)
score_train


# In[131]:


feature_names = ['酒精','苹果酸','灰','灰的碱性','镁含量','总酚','类黄酮','非黄烷类酚类',
                 '花青素','颜色强度','色调','od280/od315稀释葡萄酒','脯氨酸']
import graphviz
dot_data = tree.export_graphviz(clf
                                ,feature_names = feature_names
                                ,class_names = ["琴酒",'雪莉','贝尔摩德']
                                ,filled = True
                                ,rounded = True
                                )
graph = graphviz.Source(dot_data)
graph


# In[132]:


clf.feature_importances_


# In[133]:


[*zip(feature_names,clf.feature_importances_)]


# In[134]:


#apply返回每个测试样本所在的叶子节点的索引  只用输X
clf.apply(Xtest)


# In[144]:


#predict返回每个测试样本的分类/回归结果  只用输X
#如果数据只有一个特征  一定要使用reshape(-1，1)增维度
#如果只有一个样本和一个特征  一定要使用reshape(1,-1)来增维
clf.predict(Xtest)


# In[ ]:





# In[ ]:





# In[ ]:




