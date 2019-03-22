#!/usr/bin/env python
# coding: utf-8

# In[126]:


import numpy as np
import pandas as pd
import math


# In[40]:


#load data

data_train = np.loadtxt(open("DataTrain_Tugas3_AI.csv", "rb"), delimiter=",", skiprows=1)
data_test = np.loadtxt(open("DataTest_Tugas3_AI.csv", "rb"), delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 4, 5,))


# In[41]:


def distance(uji, latih):
    sigma = (latih[1]-uji[1])**2 + (latih[2]-uji[2])**2 + (latih[3]-uji[3])**2 + (latih[4]-uji[4])**2 + (latih[5]-uji[5])**2
    return math.sqrt(sigma)


# In[101]:


def klasifikasi(ujis, latihs, k):
    hasil = []
    for uji in ujis:
        dist = []
        label = []
        vote = []

        for latih in latihs:
            dist.append(distance(uji, latih))
            
        best = sorted(dist)[0:k]
        
        for item in best:
            label.append(latihs[dist.index(item)][6])

        vote.append(label.count(0))
        vote.append(label.count(1))
        vote.append(label.count(2))
        vote.append(label.count(3))
        
        hasil.append(vote.index(max(vote)))
          
    return hasil


# In[120]:


def cross_val(dtrain, fold):
    step = int(len(dtrain) /fold)
    labels = []
    for i in range(0, len(dtrain), step):
        label = []
        j = i + step
        test = dtrain[i:j]
        train = np.concatenate((dtrain[:i], dtrain[j:]))
        for i in range(1, 10):
            hasil = klasifikasi(test, train, i)
            boolean = (test[:,6] == np.asarray(hasil))
            label.append(boolean.sum() / boolean.size)
            
        labels.append(label)
    
    k = []
    for label in labels:
        k.append([label.index(max(label))+1, max(label)])
    print(k)


# In[121]:


cross_val(data_train, 10)


# In[122]:


hasil = klasifikasi(data_test,data_train,3)


# In[129]:


df=pd.DataFrame(hasil)
df.to_csv("TebakanTugas3.csv", index=False, header=False)

