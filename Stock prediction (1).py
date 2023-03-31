#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import random
import tensorflow as tf
from tensorflow import keras

from keras.layers import Dense,Dropout, LSTM 
from keras.models import Sequential 
from keras.layers import  Flatten
from keras.utils.np_utils import to_categorical


# In[11]:


df = pd.read_csv(r'C:\stock pred\Stocks\aa.us.txt')
df.info()


# In[12]:


df.describe()


# In[13]:


df.head()


# In[17]:


x = df.iloc[:,[1,2,3,5]]
x=x.values
y= df.iloc[:,4].values


# In[18]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)
(x_train.shape)


# In[19]:


(y_train.shape)


# In[21]:


input_shape  = x_train.shape[1]

model = Sequential()
model.add(Dense(32, activation='relu',input_shape=(input_shape,)))

model.add(Dense(64, activation='relu',input_shape=(32, )))

model.add(Dense(1,activation='linear',input_shape=(64, )))
model.summary()


# In[22]:


model.compile(optimizer='adam', loss='mean_squared_error' , metrics=['accuracy'])
model.fit(x_train,y_train,epochs=1, shuffle=True)
pred = model.predict(x_test, verbose=0)
('Mean Absolute Error: ', mean_absolute_error(y_test, pred))


# In[23]:


('Mean Squared Error: ', mean_squared_error(y_test, pred))


# In[ ]:




