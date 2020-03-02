#!/usr/bin/env python
# coding: utf-8

# ![%E5%9B%BE%E7%89%87.png](attachment:%E5%9B%BE%E7%89%87.png)

# ![%E5%9B%BE%E7%89%87.png](attachment:%E5%9B%BE%E7%89%87.png)

# In[69]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_val_score


# In[35]:


data = pd.read_csv(r'F:\My_project\mine_happiness\sklearn课堂\data\titanic_train.csv')


# In[36]:


data.info()


# In[37]:


data.head()


# In[ ]:





# In[42]:


#数据预处理与特征方程
  #删除cabin,name,ticket列
data.drop(['Cabin','Ticket','PassengerId','Name'],inplace=True,axis=1)
data


# In[43]:


#填补缺失值
data['Age'] = data['Age'].fillna(data['Age'].mean())


# In[44]:


data.info()


# In[45]:


data = data.dropna()


# In[46]:


data.head()


# In[47]:


labels = data["Embarked"].unique().tolist()
data["Embarked"] = data["Embarked"].apply(lambda x: labels.index(x))


# In[48]:


data["Sex"]=="male"


# In[50]:


(data["Sex"]=="male").astype('int')


# In[52]:


data["Sex"] = (data["Sex"]=="male").astype('int')


# In[56]:


data.loc[:,'Sex'],data.iloc[:,3]


# In[61]:


x = data.loc[:,data.columns!="Survived"]
y = data.loc[:,data.columns=="Survived"]


# In[63]:


x.shape,y.shape


# In[64]:


Xtrain,Xtest,Ytrain,Ytest = train_test_split(x,y,test_size=0.3)


# In[67]:


for i in [Xtrain,Xtest,Ytrain,Ytest]:
    i.index = range(i.shape[0])
Xtest


# In[68]:


#普通测试
clf = DecisionTreeClassifier(random_state=25)
clf = clf.fit(Xtrain,Ytrain)
score = clf.score(Xtest,Ytest)
score


# In[70]:


#交叉验证
clf = DecisionTreeClassifier(random_state=25)
score = cross_val_score(clf,x,y,cv=10).mean()
score


# In[84]:


#学习曲线
tr = []
te = []
for i in range(10):
    clf = DecisionTreeClassifier(random_state=25
                                 ,max_depth=i+1
                                 ,criterion='entropy'
                                )
    clf = clf.fit(Xtrain,Ytrain)
    score_tr = clf.score(Xtrain,Ytrain)
    score_te = clf.score(Xtest,Ytest)
    tr.append(score_tr)
    te.append(score_te)
plt.figure()
plt.plot(range(1,11),tr,color="red",label="train")
plt.plot(range(1,11),te,color="blue",label="test")
plt.xticks(range(0,11))
print('max_train:',max(tr),'\nmax_test:',max(te))
plt.legend()
plt.show()


# In[74]:


import numpy as np


# In[81]:


#网格搜索
paras = {
    "criterion":("gini","entropy")
    ,"splitter":("best","random")
    ,"max_depth":[*range(1,8)]
 #   ,"min_samples_leaf":[*range(1,50,5)]
 #   ,"min_impurity_decrease":[*np.linspace(0,0.5,50)]
}
from sklearn.model_selection import GridSearchCV
clf = DecisionTreeClassifier(random_state=25)
GS = GridSearchCV(clf,paras,cv=10)
GS = GS.fit(Xtrain,Ytrain)


# In[82]:


GS.best_params_ #返回最佳参数组合


# In[83]:


GS.best_score_ #返回GS后的最佳模型评估


# In[ ]:




