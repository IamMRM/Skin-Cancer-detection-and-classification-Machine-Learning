#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# ACCURACY 73.8 (2000, 19472)
trainbcc = np.load("train3hist_bcc.npz")['arr_0']
trainbkl = np.load("train3hist_bkl.npz")['arr_0']
trainmel = np.load("train3hist_mel.npz")['arr_0']
valbcc = np.load("train3hist_bcc_val.npz")['arr_0']
valbkl = np.load("train3hist_bkl_val.npz")['arr_0']
valmel = np.load("train3hist_mel_val.npz")['arr_0']


# In[3]:


#HAVE TO CHECK (2000, 2000)
"""trainbcc = np.load("train_hist_bcc_b10.npz")['arr_0']
trainbkl = np.load("train_hist_bkl_b10.npz")['arr_0']
trainmel = np.load("train_hist_mel_b10.npz")['arr_0']
valbcc = np.load("train_hist_bcc_b10_val.npz")['arr_0']
valbkl = np.load("train_hist_bkl_b10_val.npz")['arr_0']
valmel = np.load("train_hist_mel_b10_val.npz")['arr_0']"""
"""trainbcc = np.load("cm_bcc.npz")['arr_0']
trainbkl = np.load("cm_bkl.npz")['arr_0']
trainmel = np.load("cm_mel.npz")['arr_0']
valbcc = np.load("cm_bcc_val.npz")['arr_0']
valbkl = np.load("cm_bkl_val.npz")['arr_0']
valmel = np.load("cm_mel_val.npz")['arr_0']"""


# In[39]:


print(trainbcc.shape)
print(trainbkl.shape)
print(trainmel.shape)
print(valbcc.shape)
print(valbkl.shape)
print(valmel.shape)


# In[40]:


X = np.vstack((trainbcc,trainbkl,trainmel))
print(X.shape)


# In[41]:


Y=[0]*400
Y=Y+[1]*800
Y=Y+[2]*800
print(len(Y))


# In[42]:


from imblearn.over_sampling import SMOTE


# In[43]:


oversample = SMOTE()
X, Y = oversample.fit_resample(X, Y)


# In[44]:


print(len(Y))
print(X.shape)


# In[45]:


from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# In[46]:


oversample = SMOTE()
undersample = RandomUnderSampler()
steps = [("o", oversample), ("u", undersample)]
pipeline = Pipeline(steps=steps)
# transform the dataset
X, Y = pipeline.fit_resample(X, Y)


# In[47]:


print(len(Y))
print(X.shape)


# In[48]:


X_Val = np.vstack((valbcc,valbkl,valmel))
Y_Val=[0]*100
Y_Val=Y_Val+[1]*200
Y_Val=Y_Val+[2]*200
print(len(Y_Val))


# In[53]:


from sklearn.ensemble import RandomForestClassifier

rnd = RandomForestClassifier(n_estimators = 100)
rnd.fit(X,Y)
rnd.score(X,Y)


# In[54]:


rnd.score(X_Val,Y_Val)


# In[143]:


import random
from random import shuffle

ind_list = [i for i in range(2400)]
random.seed(0)
shuffle(ind_list)
X  = X[ind_list,:]
Y=np.array(Y)
Y = Y[ind_list]


# In[74]:


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X)
X_Test = sc_X.transform(X_Val)

from sklearn.model_selection import GridSearchCV
param_grid = { 'C':[1600, 2000, 2200],'kernel':['rbf', 'linear'],'degree':[1],'gamma': [0.001, 0.01, 0.05]}
grid = GridSearchCV(SVC(),param_grid, scoring = 'accuracy' ,verbose = 10)
grid.fit(X_Train, Y)

print(grid.best_params_)
print(grid.score(X_Test,Y_Val))
y_pred = grid.predict(X_Test)
confusion_matrix(Y_Val, y_pred)


# In[ ]:





# # NOW ADDING OTHER FEATURES

# In[75]:


import pickle
bcc_geom = np.load("bcc_geom.pkl",allow_pickle=True)
bcc_geom_val = np.load("bcc_geom_val.pkl",allow_pickle=True)
bkl_geom = np.load("bkl_geom.pkl",allow_pickle=True)
bkl_geom_val = np.load("bkl_geom_val.pkl",allow_pickle=True)
mel_geom = np.load("mel_geom.pkl",allow_pickle=True)
mel_geom_val = np.load("mel_geom_val.pkl",allow_pickle=True)


# In[76]:


bcc_glcm = np.load("bcc_glcm.pkl",allow_pickle=True)
bcc_glcm_val = np.load("bcc_glcm_val.pkl",allow_pickle=True)
bkl_glcm = np.load("bkl_glcm.pkl",allow_pickle=True)
bkl_glcm_val = np.load("bkl_glcm_val.pkl",allow_pickle=True)
mel_glcm = np.load("mel_glcm.pkl",allow_pickle=True)
mel_glcm_val = np.load("mel_glcm_val.pkl",allow_pickle=True)


# In[77]:


print(bcc_geom.shape)
print(bcc_geom_val.shape)


# In[78]:


geomXtrain = np.vstack((bcc_geom,bkl_geom,mel_geom))
geomXval   = np.vstack((bcc_geom_val,bkl_geom_val,mel_geom_val))
glcmXtrain = np.vstack((bcc_glcm,bkl_glcm,mel_glcm))
glcmXval   = np.vstack((bcc_glcm_val,bkl_glcm_val,mel_glcm_val))


# In[79]:


print(geomXtrain.shape)
print(geomXval.shape)
print(glcmXtrain.shape)
print(glcmXval.shape)


# In[80]:


geomY=[0]*400
geomY=geomY+[1]*800
geomY=geomY+[2]*800


# In[81]:


glcmY=[0]*400
glcmY=glcmY+[1]*800
glcmY=glcmY+[2]*800


# In[82]:


oversample = SMOTE()
geomXtrain, geomY = oversample.fit_resample(geomXtrain, geomY)
oversample = SMOTE()
undersample = RandomUnderSampler()
steps = [("o", oversample), ("u", undersample)]
pipeline = Pipeline(steps=steps)
# transform the dataset
geomXtrain, geomY = pipeline.fit_resample(geomXtrain, geomY)
print(geomXtrain.shape)
print(len(geomY))


# In[83]:


geomY_Val=[0]*100
geomY_Val=geomY_Val+[1]*200
geomY_Val=geomY_Val+[2]*200
print(len(geomY_Val))


# In[84]:


oversample = SMOTE()
glcmXtrain, glcmY = oversample.fit_resample(glcmXtrain, glcmY)
oversample = SMOTE()
undersample = RandomUnderSampler()
steps = [("o", oversample), ("u", undersample)]
pipeline = Pipeline(steps=steps)
# transform the dataset
glcmXtrain, glcmY = pipeline.fit_resample(glcmXtrain, glcmY)
print(glcmXtrain.shape)
print(len(glcmY))


# In[85]:


glcmY_Val=[0]*100
glcmY_Val=glcmY_Val+[1]*200
glcmY_Val=glcmY_Val+[2]*200
print(len(glcmY_Val))


# In[86]:


rnd = RandomForestClassifier(n_estimators = 100)
rnd.fit(geomXtrain,geomY)
print(rnd.score(geomXtrain,geomY))
print(rnd.score(geomXval,geomY_Val))


# In[87]:


rnd = RandomForestClassifier(n_estimators = 100)
rnd.fit(glcmXtrain,glcmY)
print(rnd.score(glcmXtrain,glcmY))
print(rnd.score(glcmXval,glcmY_Val))


# In[88]:


X_TOTAL = np.hstack((geomXtrain,glcmXtrain))
print(X_TOTAL.shape)
X_TOTALval = np.hstack((geomXval,glcmXval))
print(X_TOTALval.shape)
rnd = RandomForestClassifier(n_estimators = 100)
rnd.fit(X_TOTAL,geomY)
print(rnd.score(X_TOTAL,geomY))
print(rnd.score(X_TOTALval,geomY_Val))


# In[89]:


X_TOTAL = np.hstack((X,glcmXtrain))
print(X_TOTAL.shape)
X_TOTALval = np.hstack((X_Val,glcmXval))
print(X_TOTALval.shape)
rnd = RandomForestClassifier(n_estimators = 100)
rnd.fit(X_TOTAL,geomY)
print(rnd.score(X_TOTAL,geomY))
print(rnd.score(X_TOTALval,geomY_Val))


# In[90]:


X_TOTAL = np.hstack((X,geomXtrain))
print(X_TOTAL.shape)
X_TOTALval = np.hstack((X_Val,geomXval))
print(X_TOTALval.shape)
rnd = RandomForestClassifier(n_estimators = 100)
rnd.fit(X_TOTAL,geomY)
print(rnd.score(X_TOTAL,geomY))
print(rnd.score(X_TOTALval,geomY_Val))


# In[91]:


X_TOTAL = np.hstack((X,geomXtrain,glcmXtrain))
print(X_TOTAL.shape)
X_TOTALval = np.hstack((X_Val,geomXval,glcmXval))
print(X_TOTALval.shape)


# In[92]:


rnd = RandomForestClassifier(n_estimators = 100)
rnd.fit(X_TOTAL,geomY)
print(rnd.score(X_TOTAL,geomY))
print(rnd.score(X_TOTALval,geomY_Val))


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_TOTAL)
X_Test = sc_X.transform(X_TOTALval)

from sklearn.model_selection import GridSearchCV
param_grid = { 'C':[1600, 2000, 2200],'kernel':['rbf', 'linear'],'degree':[1],'gamma': [0.001, 0.01, 0.05]}
grid = GridSearchCV(SVC(),param_grid, scoring = 'accuracy' ,verbose = 10)
grid.fit(X_Train, geomY)

print(grid.best_params_)
print(grid.score(X_Test,geomY_Val))
y_pred = grid.predict(X_Test)
confusion_matrix(geomY_Val, y_pred)