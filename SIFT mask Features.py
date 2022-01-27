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
import pandas as pd
import csv
from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm, datasets, metrics
# In[2]:


file_name = "Code/train/les/"
file_name2 = "Code/train/ls_mask_segmentation/"
file_name3 = "Code/train/nv/"
file_name4 = "Code/train/nv_mask_segmentation/"
file_name5 = "Code/val/les/"
file_name6 = "Code/val/les_seg_val/"
file_name7 = "Code/val/nv/"
file_name8 = "Code/val/nv_seg_val/"


# In[20]:


file = helper.filee(file_name)
mask, img_rgb, image, output, grayScale = helper.loading(56,file,file_name,file_name2)
helper.plot(mask, img_rgb, image, output, grayScale)


# In[70]:


file = helper.filee(file_name3)
mask1, img_rgb1, image1, output1, grayScale1 = helper.loading(13,file,file_name3,file_name4)
helper.plot(mask1, img_rgb1, image1, output1, grayScale1)


# In[3]:


sift = cv2.xfeatures2d.SIFT_create()


# In[9]:


keypoints_1, descriptors_1 = sift.detectAndCompute(output,mask)
 
img_1 = cv2.drawKeypoints(grayScale,keypoints_1,output)
plt.imshow(img_1)


# In[10]:


dico=[]
for i,nam in enumerate(file):
    mask, img_rgb, image, output, grayScale = helper.loading(i,file,file_name,file_name2)
    keypoints_1, descriptors_1 = sift.detectAndCompute(output,mask)
    for d in descriptors_1:
        dico.append(d)


# In[15]:


file = helper.filee(file_name3)
dico2=[]
for i,nam in enumerate(file):
    mask, img_rgb, image, output, grayScale = helper.loading(i,file,file_name3,file_name4)
    #helper.plot(mask, img_rgb, image, output, grayScale)
    keypoints_1, descriptors_1 = sift.detectAndCompute(output,mask)
    for d in descriptors_1:
        dico2.append(d)


# In[17]:


#FOR LOADING 
"""dico= np.load("lesSIFT.npz")
print(dico['arr_0'].shape)
dico2 = np.load("nvSIFT.npz")
print(dico2['arr_0'].shape)
fini=list(dico['arr_0'])+list(dico2['arr_0'])"""


# In[22]:


print(len(dico))
print(len(dico2))
fini=dico+dico2
print(len(fini))


# In[21]:


noClasses = 2
k = np.size(noClasses)*10
batch_size = np.size(len(file)*2) * 3
print(batch_size)
print(k)
kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1).fit(fini)


# In[28]:


kmeans.verbose = False
histo_list = []
file = helper.filee(file_name)
for abc in range(2):
    for i,nam in enumerate(file):
        mask, __ , __ , output, __ = helper.loading(i,file,file_name,file_name2)
        keypoints_1, descriptors_1 = sift.detectAndCompute(output,mask)
        histo = np.zeros(k)
        nkp = np.size(keypoints_1)

        for d in descriptors_1:
            idx = kmeans.predict([d])
            histo[idx] += 1/nkp # Because we need normalized histograms
        print("Done with "+str(i))
        histo_list.append(histo)
    print("ONE DONE NOW NEXT")
    file = helper.filee(file_name3)
    file_name=file_name3
    file_name2=file_name4


# In[96]:

X = np.array(histo_list)

#FOR LOADING
"""X= np.load("histSIFT.npz")
print(X['arr_0'].shape)"""


# In[23]:


one = [1]*2400
zero = [0]*2400
Y = zero+one
print(len(Y))


# In[25]:


mlp = MLPClassifier(verbose=True, max_iter=600000)
mlp.fit(X['arr_0'],Y)


# In[43]:


mlp.score(X['arr_0'],Y)
mlp.score(val_X,valY)


# In[33]:




rnd = RandomForestClassifier(n_estimators = 100)
rnd.fit(X['arr_0'],Y)
rnd.score(X['arr_0'],Y)#Training score = 1


# In[27]:


file_name5 = "Code/val/les/"
file_name6 = "Code/val/les_seg_val/"
file_name7 = "Code/val/nv/"
file_name8 = "Code/val/nv_seg_val/"
file = helper.filee(file_name5)
dico3=[]
for i,nam in enumerate(file):
    mask, __, __, output, __ = helper.loading(i,file,file_name5,file_name6)
    keypoints_1, descriptors_1 = sift.detectAndCompute(output,mask)
    for d in descriptors_1:
        dico3.append(d)
print(len(dico3))
file = helper.filee(file_name7)
dico4=[]
for i,nam in enumerate(file):
    mask, __, __, output, __ = helper.loading(i,file,file_name7,file_name8)
    keypoints_1, descriptors_1 = sift.detectAndCompute(output,mask)
    for d in descriptors_1:
        dico4.append(d)
print(len(dico4))


# In[29]:


np.savez("vallesSIFT", np.array(dico3))
np.savez("valnvSIFT", np.array(dico4))


# In[31]:


val_fini=dico3+dico4
#NOW FIND HISTOGRAM


# In[36]:


kmeans.verbose = False
file = helper.filee(file_name5)
histo_list=[]
for abc in range(2):
    for i,nam in enumerate(file):
        mask, __ , __ , output, __ = helper.loading(i,file,file_name5,file_name6)
        keypoints_1, descriptors_1 = sift.detectAndCompute(output,mask)
        histo = np.zeros(k)
        nkp = np.size(keypoints_1)

        for d in descriptors_1:
            idx = kmeans.predict([d])
            histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly
        histo_list.append(histo)
    print("ONE DONE NOW NEXT")
    file = helper.filee(file_name7)
    file_name5=file_name7
    file_name6=file_name8


# In[37]:


val_X = np.array(histo_list)
print(val_X.shape)
np.savez("valhistSIFT", val_X)


# In[40]:


one = [1]*600
zero = [0]*600
valY = zero+one
print(len(valY))


# In[41]:


rnd.score(val_X,valY)


# In[47]:



sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X['arr_0'])
X_Test = sc_X.transform(val_X)
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X['arr_0'], Y)

# Predicting the test set results

Y_Pred = classifier.predict(val_X)

# Making the Confusion Matrix 


cm = confusion_matrix(valY, Y_Pred)
print(accuracy_score(valY, Y_Pred))

# In[52]:


y_score=rnd.predict_proba(val_X)[:, 1]
print("Classification report for classifier %s:\n%s\n"
    % (rnd, metrics.classification_report(valY, y_score>.5)))