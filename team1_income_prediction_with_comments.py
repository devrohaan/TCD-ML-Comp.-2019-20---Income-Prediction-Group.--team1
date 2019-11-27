#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


# In[ ]:


def create_feat_count(df,feat):
    feat_count = df.groupby([feat]).size().reset_index()
    feat_count.columns = [feat,'%s_count'%(feat)]
    df = df.merge(feat_count,how='left',on=[feat])
    return df


# In[ ]:


# Reading in the test and training data
train = pd.read_csv("group-income-train.csv")
test = pd.read_csv("group-income-test.csv")
data = pd.concat([train,test],ignore_index=True)


# In[2]:


# Coverting Additional Income to Ints for Easier Processing
data['Yearly Income in addition to Salary (e.g. Rental Income)'] = data['Yearly Income in addition to Salary (e.g. Rental Income)'].map(lambda x:x.replace(' EUR',''))
data['Yearly Income in addition to Salary (e.g. Rental Income)'] = data['Yearly Income in addition to Salary (e.g. Rental Income)'].astype(float)
data['Yearly Income in addition to Salary (e.g. Rental Income)']=data['Yearly Income in addition to Salary (e.g. Rental Income)'].astype(int)                            


# In[ ]:


# Label Encoding all Necessary Columns for Processing along with creating a feature count col
cols = data.columns.tolist()
feat_cols = [col for col in data.columns if col not in ['Instance','Total Yearly Income [EUR]']]
for col in feat_cols:
    data = create_feat_count(data,col)
feat_cols = [col for col in data.columns if col not in ['Instance','Total Yearly Income [EUR]']]
obj_col = data[feat_cols].dtypes[data[feat_cols].dtypes == 'object'].index.tolist()
for col in obj_col:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))


# In[ ]:


# Splitting our nicely formatted data back into test and training sets
train = data[data['Total Yearly Income [EUR]'].notnull()]
test = data[data['Total Yearly Income [EUR]'].isnull()]    


# In[ ]:


# This messy pip install was for convenience when running on Google Colab/AWS Sagemaker:
get_ipython().system('pip install lightgbm')
import lightgbm as lgb

# Running a k-fold cross validation as in:
# https://machinelearningmastery.com/k-fold-cross-validation/
# Using tweedie distribution with gdbt boosting
params = {
          'max_depth': 30,
          'learning_rate': 0.02,
          "boosting": "gbdt",
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          'objective':'tweedie',
          'gpu_platform_id': 0,
          'gpu_device_id': 0,
          'num_iterations' : 200000,
         }
# N-folds opted for 5, according to researched material online 5 or 10 can be ideal for this process
folds = 5
seed = 2019
pre_sub = pd.DataFrame()
kf = StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
ix = 0
for tr_idx,val_idx in kf.split(train,train['Country']):
    x_train,y_train = train[feat_cols].iloc[tr_idx],train['Total Yearly Income [EUR]'].iloc[tr_idx]
    x_val,y_val = train[feat_cols].iloc[val_idx],train['Total Yearly Income [EUR]'].iloc[val_idx]
    trn_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_val, label=y_val)
    # 15000 Redundant now as overridden with num_iterations
    clf = lgb.train(params, trn_data, 15000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)
    test_pre = clf.predict(test[feat_cols])
    pre_sub[ix] = test_pre
    ix += 1
'done'


# In[ ]:


# Getting the mean of 5-fold cross validation and using as answer
pre_sub['sum'] = pre_sub[[0,1,2,3,4]].mean(axis=1)
pre_sub.head()


# In[ ]:


# Printing resolves to CSV
sub = pd.DataFrame()
sub['Instance'] = test['Instance'].tolist()
sub['Total Yearly Income [EUR]'] = pre_sub['sum'].values
sub.to_csv("awssubmission.csv",index=False)
'done'


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




