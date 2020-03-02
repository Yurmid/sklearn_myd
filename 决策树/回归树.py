#!/usr/bin/env python
# coding: utf-8

# In[1]:


#几乎所有参数、属性和接口都和分类树一摸一样，需要注意的是，在回归树当中
#没有标签分布是否均衡的问题，应为没有class_weigh这种参数。
# 分类树的criterion不是不纯度了，而是mse(父节点与子节点之间均方误差的差额，用叶子节点的均值来最小化L1损失)、
#                                  friedman_mse(改进分支问题后的均方误差)
#                                  mae(绝对平均误差，使用叶结点的中值来最小化L1损失)
# mse是分枝质量衡量指标，也是最重要的衡量回归树回归质量的指标，！！但是score默认返回的是R方(1-残差平方和/总平方和)，不是MSE。


# In[140]:


from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score,train_test_split
import pandas as pd
import numpy as np


# In[21]:


boston = load_boston()
boston.data


# In[22]:


boston.data.shape,boston.feature_names


# In[146]:


boston.target,boston.target.shape


# In[25]:


data_col = pd.concat([pd.DataFrame(boston.data),pd.DataFrame(boston.target)],axis=1)
data_col


# In[123]:


#测试state的选择
randoms_state = []
state_node = 5.0
for i in range(50):
    #实例化回归树
    regressor = DecisionTreeRegressor(random_state=i) 
    #交叉验证
    a=cross_val_score(regressor #实例化后的算法模型，任何都可以
                    ,boston.data #不需要划分训练集和测试集的训练data
                    ,boston.target #不需要划分...的完整训练label
                    ,cv=10 #交叉验证划分成几部分
                    ,scoring='neg_mean_squared_error' #不填的话默认返回R方(越接近1越好)，一般都用负均方误差(绝对值越小越好)
                   )
    k =abs(sum(a)/100)
    randoms_state.append(k)
    if k < state_node:
        state_node = k
        min_state = i
plt.plot(range(0,50),randoms_state,color='red',label="random_state")
plt.show()
print(min_state,state_node)


# In[230]:


##测试cv的选择
#randoms_cv = []
#cv_node = 1
#for i in range(2,50):
#    #实例化回归树
#    regressor = DecisionTreeRegressor(random_state=46) 
#    #交叉验证
#    a=cross_val_score(regressor #实例化后的算法模型，任何都可以
#                    ,boston.data #不需要划分训练集和测试集的训练data
#                    ,boston.target #不需要划分...的完整训练label
#                    ,cv=i #交叉验证划分成几部分
#                    ,scoring='neg_mean_squared_error' #不填的话默认返回R方(越接近1越好)，一般都用负均方误差(绝对值越小越好)
#                   )
#    k =abs(sum(a)/100)/i
#    randoms_cv.append(k)
#    if k < cv_node:
#        cv_node = k
#        min_cv = i
#plt.plot(range(2,50),randoms_cv,color='red',label="random_cv")
#plt.show()
#print(min_cv,cv_node)
#


# In[190]:


Xtrain,Xtest,Ytrain,Ytest = train_test_split(boston.data,boston.target,test_size=0.3)


# In[191]:


Xtrain.shape,Ytrain.shape


# In[1]:


regressor = DecisionTreeRegressor(random_state=37
                                 ,max_depth=5
                                 ,min_samples_split=10
                                 ) 


# In[261]:


regressor.fit(Xtrain,Ytrain)
regressor.score(Xtrain,Ytrain)
regressor.score(Xtest,Ytest)


# In[227]:


[*zip(feature_names,regressor.feature_importances_)]


# In[ ]:


#   import graphviz
#   dot_data = tree.export_graphviz(regressor
#                                 #  ,feature_names = feature_names
#                                 #  ,class_names = 
#                                   ,filled = True
#                                   ,rounded = True
#                                   )
#   graph = graphviz.Source(dot_data)
#   graph


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[75]:


#生成正弦曲线，并加噪声点
import matplotlib.pyplot as plt
rng = np.random.RandomState(1)  #生成随机数种子，一样的模式生成随机数 
x = np.sort(5*rng.rand(80,1),axis=0)  #rand随机生成0-1之间的随机数，一个参数是生成参数个随机数，两个参数生成m*n维矩阵
                                      #*5是变成0-5之间，sort是从小到大排序，axis是排序方向，排行。
y = np.sin(x).ravel()                 #ravel()降维函数(降1维)  一维数组不分行列。
y[::5] += 3*(0.5-rng.rand(len(y[::5])))


# In[76]:


plt.figure()
plt.scatter(x,y,s=20,edgecolor='black',c='darkorange',label='data')
plt.show


# In[90]:


regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(x,y)
regr_2.fit(x,y)


# In[91]:


x_test = np.arange(0.0,5.0,0.01)[:,np.newaxis] #newaxis为增维切片
y_1 = regr_1.predict(x_test)
y_2 = regr_2.predict(x_test)


# In[92]:


plt.figure()
plt.scatter(x,y,s=20,edgecolor='black',c='darkorange',label='data')
plt.plot(x_test,y_1,color='cornflowerblue',label='max_depth=2',linewidth=2)
plt.plot(x_test,y_2,color='yellowgreen',label='max_depth=5',linewidth=2)
plt.xlabel('data')
plt.ylabel('target')
plt.title('Decision Tree Regression')
plt.legend() #显示图例
plt.show()


# In[114]:


ls =-1.32412
abs(ls)


# In[60]:


regr_2.score(x,y)


# In[ ]:




