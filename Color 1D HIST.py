#!/usr/bin/env python
# coding: utf-8

# In[94]:


import numpy as np
import cv2
from skimage import measure
from skimage.measure import label, regionprops
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 70,10
import helper
import random
import pylab as pl
from sklearn.metrics import confusion_matrix,accuracy_score



# In[2]:


file_name = "Code/train/les/"
file_name2 = "Code/train/ls_mask_segmentation/"
file_name3 = "Code/train/nv/"
file_name4 = "Code/train/nv_mask_segmentation/"
file_name5 = "Code/val/les/"
file_name6 = "Code/val/les_seg_val/"
file_name7 = "Code/val/nv/"
file_name8 = "Code/val/nv_seg_val/"
file = helper.filee(file_name)


# In[3]:


mask, img_rgb, image, output, grayScale = helper.loading(54,file,file_name,file_name2)
helper.plot(mask, img_rgb, image, output, grayScale)


bins = 30
train_hist=[]
file = helper.filee(file_name)
for i,nam in enumerate(file):
    temp=np.zeros(shape=(7,bins,1))
    mask, img_rgb, image, output, grayScale= helper.loading(i,file,file_name,file_name2)
    imgHSV = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    temp[0] = cv2.calcHist([img_rgb],channels=[0],mask=mask,histSize=[bins],ranges=[0,255])
    temp[1] = cv2.calcHist([img_rgb],channels=[1],mask=mask,histSize=[bins],ranges=[0,255])
    temp[2] = cv2.calcHist([img_rgb],channels=[2],mask=mask,histSize=[bins],ranges=[0,255])
    temp[3] = cv2.calcHist([imgHSV],channels=[0],mask=mask,histSize=[bins],ranges=[0,180])
    temp[4] = cv2.calcHist([imgHSV],channels=[1],mask=mask,histSize=[bins],ranges=[0,255])
    temp[5] = cv2.calcHist([imgHSV],channels=[2],mask=mask,histSize=[bins],ranges=[0,255])
    temp[6] = cv2.calcHist([grayScale],channels=[0],mask=mask,histSize=[bins],ranges=[0,255])
    temp=temp.flatten()
    train_hist.append(temp)

print(temp.shape)
#print(temp.flatten().shape)
temp=temp.flatten()

# In[39]:


file = helper.filee(file_name3)
for i,nam in enumerate(file):
    temp=np.zeros(shape=(7,bins,1))
    mask, img_rgb, image, output, grayScale= helper.loading(i,file,file_name3,file_name4)
    imgHSV = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    temp[0] = cv2.calcHist([img_rgb],channels=[0],mask=mask,histSize=[bins],ranges=[0,255])
    temp[1] = cv2.calcHist([img_rgb],channels=[1],mask=mask,histSize=[bins],ranges=[0,255])
    temp[2] = cv2.calcHist([img_rgb],channels=[2],mask=mask,histSize=[bins],ranges=[0,255])
    temp[3] = cv2.calcHist([imgHSV],channels=[0],mask=mask,histSize=[bins],ranges=[0,180])
    temp[4] = cv2.calcHist([imgHSV],channels=[1],mask=mask,histSize=[bins],ranges=[0,255])
    temp[5] = cv2.calcHist([imgHSV],channels=[2],mask=mask,histSize=[bins],ranges=[0,255])
    temp[6] = cv2.calcHist([grayScale],channels=[0],mask=mask,histSize=[bins],ranges=[0,255])
    temp=temp.flatten()
    train_hist.append(temp)


# In[40]:


Y=[0]*2400
Y=Y+[1]*2400
X = np.array(train_hist)


# In[41]:


print(X.shape)


val_hist=[]
file = helper.filee(file_name5)
for i,nam in enumerate(file):
    temp=np.zeros(shape=(7,bins,1))
    mask, img_rgb, image, output, grayScale= helper.loading(i,file,file_name5,file_name6)
    imgHSV = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    temp[0] = cv2.calcHist([img_rgb],channels=[0],mask=mask,histSize=[bins],ranges=[0,255])
    temp[1] = cv2.calcHist([img_rgb],channels=[1],mask=mask,histSize=[bins],ranges=[0,255])
    temp[2] = cv2.calcHist([img_rgb],channels=[2],mask=mask,histSize=[bins],ranges=[0,255])
    temp[3] = cv2.calcHist([imgHSV],channels=[0],mask=mask,histSize=[bins],ranges=[0,180])
    temp[4] = cv2.calcHist([imgHSV],channels=[1],mask=mask,histSize=[bins],ranges=[0,255])
    temp[5] = cv2.calcHist([imgHSV],channels=[2],mask=mask,histSize=[bins],ranges=[0,255])
    temp[6] = cv2.calcHist([grayScale],channels=[0],mask=mask,histSize=[bins],ranges=[0,255])
    temp=temp.flatten()
    val_hist.append(temp)


# In[45]:


file = helper.filee(file_name7)
for i,nam in enumerate(file):
    temp=np.zeros(shape=(7,bins,1))
    mask, img_rgb, image, output, grayScale= helper.loading(i,file,file_name7,file_name8)
    imgHSV = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    temp[0] = cv2.calcHist([img_rgb],channels=[0],mask=mask,histSize=[bins],ranges=[0,255])
    temp[1] = cv2.calcHist([img_rgb],channels=[1],mask=mask,histSize=[bins],ranges=[0,255])
    temp[2] = cv2.calcHist([img_rgb],channels=[2],mask=mask,histSize=[bins],ranges=[0,255])
    temp[3] = cv2.calcHist([imgHSV],channels=[0],mask=mask,histSize=[bins],ranges=[0,180])
    temp[4] = cv2.calcHist([imgHSV],channels=[1],mask=mask,histSize=[bins],ranges=[0,255])
    temp[5] = cv2.calcHist([imgHSV],channels=[2],mask=mask,histSize=[bins],ranges=[0,255])
    temp[6] = cv2.calcHist([grayScale],channels=[0],mask=mask,histSize=[bins],ranges=[0,255])
    temp=temp.flatten()
    val_hist.append(temp)


# In[46]:


Y_Val=[0]*600
Y_Val=Y_Val+[1]*600
X_Val = np.array(val_hist)


# In[47]:


print(X_Val.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[64]:


file = helper.filee(file_name3)
for i,nam in enumerate(file):
    mask, img_rgb, __, __, __ = helper.loading(i,file,file_name3,file_name4)
    imgHSV = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    histg = cv2.calcHist([imgHSV],channels=[0,1],mask=mask,ranges=[0,180,0,255],histSize=[30,32])
    train_hist.append(histg)


# In[ ]:





# In[82]:


val_hist=[]
file = helper.filee(file_name5)
for i,nam in enumerate(file):
    mask, img_rgb, __, __, __ = helper.loading(i,file,file_name5,file_name6)
    imgHSV = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    histg = cv2.calcHist([imgHSV],channels=[0,1],mask=mask,ranges=[0,180,0,255],histSize=[30,32])
    val_hist.append(histg)


# In[83]:


file = helper.filee(file_name7)
for i,nam in enumerate(file):
    mask, img_rgb, __, __, __ = helper.loading(i,file,file_name7,file_name8)
    imgHSV = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    histg = cv2.calcHist([imgHSV],channels=[0,1],mask=mask,ranges=[0,180,0,255],histSize=[30,32])
    val_hist.append(histg)


# In[88]:


X_Val = []
for a in val_hist:
    X_Val.append(np.array(a).flatten())
Y_Val=[0]*600
Y_Val=Y_Val+[1]*600


# In[94]:
from sklearn.ensemble import RandomForestClassifier

rnd = RandomForestClassifier(n_estimators = 100)
rnd.fit(X,Y)
rnd.score(X,Y)


# In[58]:


rnd.score(X_Val,Y_Val)


# In[91]:


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X)
X_Test = sc_X.transform(X_Val)

from sklearn.model_selection import GridSearchCV
param_grid = { 'C':[1600, 2000, 2200],'kernel':['rbf'],'degree':[1],'gamma': [0.001, 0.01, 0.05]}
grid = GridSearchCV(SVC(),param_grid, scoring = 'accuracy' ,verbose = 10)
grid.fit(X_Train, Y)

print(grid.best_params_)
print(grid.score(X_Test,Y_Val))
y_pred = grid.predict(X_Test)
confusion_matrix(Y_Val, y_pred)

rnd.score(X_Val,Y_Val)