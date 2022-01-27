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


# In[4]:


imgHSV = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
helper.plot(mask, img_rgb, image, imgHSV, grayScale)


# In[63]:


train_hist=[]
file = helper.filee(file_name)
for i,nam in enumerate(file):
    mask, img_rgb, __, __, grayScale = helper.loading(i,file,file_name,file_name2)
    imgHSV = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    histhsv = cv2.calcHist([imgHSV],channels=[0,1,2],mask=mask,ranges=[0,180,0,255,0,255],histSize=[30,32,16])
    histrgb = cv2.calcHist([img_rgb],channels=[0,1,2],mask=mask,ranges=[0,255,0,255,0,255],histSize=[16,16,16])
    gray = cv2.calcHist([grayScale], channels=[0], mask=mask, histSize=[16], ranges=[0,255])
    
    train_hist.append(np.hstack((histrgb.flatten(), histhsv.flatten(),gray.flatten())))


# In[64]:


file = helper.filee(file_name3)
for i,nam in enumerate(file):
    mask, img_rgb, __, __, grayScale = helper.loading(i,file,file_name3,file_name4)
    imgHSV = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    histhsv = cv2.calcHist([imgHSV],channels=[0,1,2],mask=mask,ranges=[0,180,0,255,0,255],histSize=[30,32,16])
    histrgb = cv2.calcHist([img_rgb],channels=[0,1,2],mask=mask,ranges=[0,255,0,255,0,255],histSize=[16,16,16])
    gray = cv2.calcHist([grayScale], channels=[0], mask=mask, histSize=[16], ranges=[0,255])
    
    train_hist.append(np.hstack((histrgb.flatten(), histhsv.flatten(),gray.flatten())))


# In[91]:


Y=[0]*2400
Y=Y+[1]*2400
X = np.array(train_hist)


# In[81]:


#np.savez("trainCOLORhist", X)


# In[14]:


from sklearn.ensemble import RandomForestClassifier

rnd = RandomForestClassifier(n_estimators = 100)
rnd.fit(X,Y)
rnd.score(X,Y)


# In[4]:


val_hist=[]
file = helper.filee(file_name5)
for i,nam in enumerate(file):
    mask, img_rgb, __, __, grayScale = helper.loading(i,file,file_name5,file_name6)
    imgHSV = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    histhsv = cv2.calcHist([imgHSV],channels=[0,1,2],mask=mask,ranges=[0,180,0,255,0,255],histSize=[30,32,16])
    histrgb = cv2.calcHist([img_rgb],channels=[0,1,2],mask=mask,ranges=[0,255,0,255,0,255],histSize=[16,16,16])
    gray = cv2.calcHist([grayScale], channels=[0], mask=mask, histSize=[16], ranges=[0,255])
    
    val_hist.append(np.hstack((histrgb.flatten(), histhsv.flatten(),gray.flatten())))


# In[5]:


file = helper.filee(file_name7)
for i,nam in enumerate(file):
    mask, img_rgb, __, __, __ = helper.loading(i,file,file_name7,file_name8)
    imgHSV = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    histhsv = cv2.calcHist([imgHSV],channels=[0,1,2],mask=mask,ranges=[0,180,0,255,0,255],histSize=[30,32,16])
    histrgb = cv2.calcHist([img_rgb],channels=[0,1,2],mask=mask,ranges=[0,255,0,255,0,255],histSize=[16,16,16])
    gray = cv2.calcHist([grayScale], channels=[0], mask=mask, histSize=[16], ranges=[0,255])
    
    val_hist.append(np.hstack((histrgb.flatten(), histhsv.flatten(),gray.flatten())))


# In[6]:


X_Val=np.array(val_hist)
Y_Val=[0]*600
Y_Val=Y_Val+[1]*600


# In[11]:


#print(len(X_Val))
#print(X_Val[0].shape)


# In[12]:


#np.savez("valCOLORhist", X_Val)


# In[13]:


rnd.score(X_Val,Y_Val)


# In[ ]:


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


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
model_params = {
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10,20]
        }
    },
    'mlp':{
        'model':MLPClassifier(verbose=True, max_iter=600000),
        'params':{
            'activation':['identity', 'logistic', 'tanh', 'relu'],
            'solver':['lbfgs', 'sgd', 'adam'],
            'hidden_layer_sizes':[100,150,200,250]
        }
    },
    'xgboost':{
        'model':XGBClassifier(verbosity=0,num_class=2),
        'params':{'booster':['gbtree','dart'],
        'eta':[0.3,0.5],
        'max_depth':[5,6],
        'objective':['multi:softmax'],
        'n_estimators': [50,150]
        }
    }
}


# In[ ]:


scores = []
from sklearn.model_selection import GridSearchCV
for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(X, Y)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df