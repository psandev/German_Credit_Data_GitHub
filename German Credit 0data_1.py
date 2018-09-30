
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pylab as plt
# get_ipython().run_line_magic('matplotlib', 'notebook')
# get_ipython().run_line_magic('matplotlib', 'notebook')


# In[4]:


data = pd.read_csv('german.data', sep=' ')
data.head()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.isnull().sum()


# In[8]:


data.nunique()


# In[9]:


cat_vars = list(data.columns)
cat_vars.remove('1169')
cat_vars.remove('1.1')
print(cat_vars)
cat_vars_2_levels = data.columns[data.nunique()<3]
cat_vars_more_levels = data.columns[data.nunique()>2]
print(cat_vars_2_levels)
print(cat_vars_more_levels)


# In[10]:


data[cat_vars] = data[cat_vars].apply(lambda x: x.astype('category'))
data['1169'] = data['1169'].astype(np.int16)
data.info()


# In[11]:


# data[cat_vars_2_levels] = data[cat_vars_2_levels].apply(lambda x: x.cat.codes)
# data.dtypes


# In[12]:


data.head(10)


# In[13]:



from sklearn.preprocessing import  StandardScaler
scaler = StandardScaler()
data['1169'] = scaler.fit_transform(data['1169'].values.reshape(-1,1))
data['1169'].plot.hist()


# In[14]:


data['1.1'].value_counts()


# In[15]:


from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
X = data.drop('1.1', axis=1)
y = data['1.1']
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.7, random_state=1234)


# In[16]:


y.head()


# In[17]:


X_train.info()


# In[18]:


categorical_features_indices = np.where(X.dtypes != np.float)[0]
categorical_features_indices


model=CatBoostClassifier(iterations=500, depth=3, learning_rate=0.1, loss_function='Recall')
# In[19]:


model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_validation, y_validation),plot=True)
# print(categorical_features_indices)

