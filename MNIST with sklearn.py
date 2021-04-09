#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


# In[2]:


data=fetch_openml('mnist_784')


# In[3]:


x,y=data['data'],data['target']


# In[4]:


x.ndim


# In[5]:


y.ndim


# In[6]:


import matplotlib


# In[7]:


some_digit=x[3600]
image_digit=some_digit.reshape(28,28)


# In[8]:


plt.imshow(image_digit)


# In[9]:


y[3600]


# In[10]:


x_train=x[:60000]
x_test=x[60000:]
y_train=y[:60000]
y_test=y[60000:]


# In[11]:


import numpy as np
shuffle_data=np.random.permutation(60000)
x_train=x_train[shuffle_data]
y_train=y_train[shuffle_data]


# In[12]:


y_train=y_train.astype(np.int8)
y_test=y_test.astype(np.int8)
y_train_8=(y_train==8)
y_test_8=(y_test==8)


# In[13]:


y_test_8


# In[14]:


y_train_8


# In[15]:


from sklearn.linear_model import LogisticRegression 


# In[16]:


classifier= LogisticRegression(tol=0.1)


# In[17]:


classifier.fit(x_train,y_train_8)


# In[18]:


classifier.predict([some_digit])


# In[19]:


from sklearn.model_selection import cross_val_score
k=cross_val_score(classifier,x_train,y_train_8,cv=3, scoring='accuracy')


# In[20]:


k.mean()*100 #94% accurate


# ## not 2

# In[21]:


y_train=y_train.astype(np.int8)
y_test=y_test.astype(np.int8)
y_train_n8=(y_train!=8)
y_test_n8=(y_test!=8)


# In[22]:


y_train_n8


# In[23]:


classifier.fit(x_train,y_train_n8)


# In[24]:


classifier.predict([some_digit])


# In[25]:


from sklearn.model_selection import cross_val_score
k=cross_val_score(classifier,x_train,y_train_n8,cv=3, scoring='accuracy')


# In[26]:


k.mean()*100

