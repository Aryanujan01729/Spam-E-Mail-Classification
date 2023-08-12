#!/usr/bin/env python
# coding: utf-8

# In[115]:


get_ipython().system('pip install imbalanced-learn')


# In[144]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import keras
from keras.layers import Dropout 
from sklearn.model_selection import train_test_split
from collections import Counter
import re
from imblearn.over_sampling import SMOTE


# In[10]:


Dataset=pd.read_csv('emails.csv')


# In[75]:


len(Dataset)


# In[12]:


Dataset_1=Dataset.drop('Prediction',axis=1)
Dataset_1=Dataset_1.drop('Email No.',axis=1)
Dataset_1.head()


# In[105]:


y1=Dataset.loc[:,'Prediction']


# In[15]:


y.head()


# In[16]:


x=Dataset.loc[0]


# In[17]:


print(x)


# In[369]:


model = keras.Sequential([
    keras.layers.Input(shape=(3000,)),
    keras.layers.Dense(100, activation='relu',kernel_regularizer=keras.regularizers.l1(0.02)),
    keras.layers.Dense(50, activation='relu',kernel_regularizer=keras.regularizers.l1(0.001)),
    Dropout(0.2),
    keras.layers.Dense(25, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(1, activation='sigmoid') 
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


# In[370]:


X = Dataset_1.to_numpy()
y = y1.to_numpy()


# In[372]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[444]:


sm=SMOTE(random_state=10, sampling_strategy=1.0)
X_train_res,y_train_res=sm.fit_resample(X_train,y_train)


# In[445]:


n=0
p=0
for i in y_train_res:
    if i==1:
        p=p+1
    else:
        n=n+1


# In[446]:


n


# In[447]:


model.fit(X_train_res,y_train_res, epochs=150)


# In[448]:


loss, accuracy = model.evaluate(X_test, y_test)


# In[459]:


E_mail=input()


# In[460]:


Lower_case=E_mail.lower()


# In[461]:


arr=[]
words = re.findall(r'\w+',Lower_case )
word_counter = Counter(words)
words=Dataset_1.columns
for word in words:
    frequency = word_counter.get(word,0)
    arr.append(frequency)


# In[463]:


arr=pd.DataFrame(arr)


# In[464]:


word_para= arr.to_numpy()


# In[465]:


word_para=word_para.reshape(-1,3000)


# In[466]:


a=model.predict(word_para)


# In[467]:


if a<0.1:
    print("E-Mail is not Spam")
else:
    print("E-Mail is Spam")


# In[468]:


a


# In[ ]:





# In[ ]:




